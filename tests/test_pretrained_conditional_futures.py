"""Conditional future sampling, registration parity, and causal input checks."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import evaluate_pretrained_conditional_futures as trial
from tools import evaluate_pretrained_decision_fidelity as fidelity
from test_urban_dreamer_event_session import synthetic_network
from test_train_component_world_model import small_model
from urban_dreamer.learning.action_events import ActionEvent,pack_action_events
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import TrafficTopology,VehicleDemand


def test_private_future_sampler_is_source_sampler_on_remaining_interval():
    law=PoissonDemand(((0,0,.1),(1,0,.2)),1800.)
    before=fidelity.fingerprint(fidelity.external_rng_state())
    future,meta=trial.sample_future(law,boundary=300,seed=810001)
    actual=PoissonDemand(law.od_rates,1500.).sample(seed=810001,episode_index=0)
    assert [(v.origin_index,v.destination_index,v.departure_second) for v in future] == [
        (v.origin_index,v.destination_index,300+v.departure_second) for v in actual]
    assert all(300<v.departure_second<1800 for v in future)
    assert all(v.vehicle_id==f'future_v{i}' for i,v in enumerate(future))
    assert meta['expected_count']==pytest.approx(450.)
    assert sum(row['count'] for row in meta['od_counts'])==len(future)
    assert meta['initial_private_rng']!=meta['final_private_rng']
    assert trial.sample_future(law,boundary=300,seed=810001)==(future,meta)
    assert trial.sample_future(law,boundary=300,seed=810002)[1]['realization_sha256']!=meta['realization_sha256']
    assert fidelity.fingerprint(fidelity.external_rng_state())==before


def test_past_only_prefix_uses_actual_source_draw_without_registering_futures():
    law=PoissonDemand(((0,0,.2),),1800.)
    all_rows=law.sample(seed=620001,episode_index=0)
    past=trial.past_only_demand(law,seed=620001,boundary=900)
    assert past==tuple(row for row in all_rows if row.departure_second<900)
    assert len(past)<len(all_rows)


class FakeConnection:
    def __init__(self):
        self.routes={};self.added=[];self.type_calls=[]
        self.route=SimpleNamespace(add=self.add_route,getIDList=lambda:tuple(self.routes))
        self.vehicle=SimpleNamespace(add=lambda *a,**kw:self.added.append((a,kw)))
        self.vehicletype=SimpleNamespace(**{name:(lambda *a,key=name:self.type_calls.append((key,a)))
            for name in ('copy','setMaxSpeed','setSpeedFactor','setSpeedDeviation')})
    def add_route(self,name,edges):
        self.routes[name]=list(edges)


def make_session(vehicles,*,forbid_revisit=False):
    net=synthetic_network()
    topology=TrafficTopology(net,('a',),('d',),tuple(vehicles))
    config=SimpleNamespace(run=SimpleNamespace(seed=42),simulation=SimpleNamespace(
        vehicle_max_speed_mps=40.,sumo_extra_args=(),incident_edge='',incident_start_seconds=0.,
        incident_end_seconds=0.,incident_speed_mps=.1,episode_wall_timeout_seconds=300.,traffic_model='meso'))
    prepared=SimpleNamespace(network_xml='unused.xml')
    session=SumoEventSession(config,prepared,topology,backend='libsumo',forbid_revisit=forbid_revisit)
    session._state=lambda:{}
    return session


@pytest.mark.parametrize('forbid_revisit',[False,True])
def test_registration_matches_actual_production_reset_loop(monkeypatch,forbid_revisit):
    libsumo=pytest.importorskip('libsumo')
    vehicles=(VehicleDemand('future_v0',0,0,31.25),VehicleDemand('future_v1',0,0,45.5))
    original=FakeConnection()
    monkeypatch.setattr(libsumo,'start',lambda args:None)
    for name in ('route','vehicle','vehicletype'):
        monkeypatch.setattr(libsumo,name,getattr(original,name))
    baseline=make_session(vehicles,forbid_revisit=forbid_revisit)
    baseline.reset()  # Execute the production implementation, not a copied expected template.
    injected=make_session((),forbid_revisit=forbid_revisit)
    connection=FakeConnection()
    injected._connection=connection
    injected._time_ms=30000
    injected._flow_cache=(30000,np.ones((4,8)))
    before=fidelity.fingerprint(dict(metrics=injected.metrics,reward=injected.reward_accounting,
        active=injected._active,pending=injected._pending))
    records=trial.register_future(injected,vehicles)
    assert connection.routes==original.routes and connection.added==original.added
    assert injected._vehicles==baseline._vehicles
    assert injected._commanded_routes==baseline._commanded_routes
    assert injected._route_cursors==baseline._route_cursors
    assert injected._actual_routes==baseline._actual_routes=={}
    assert not connection.type_calls  # Existing urban_cav type is reused.
    assert injected._flow_cache is None
    assert fidelity.fingerprint(dict(metrics=injected.metrics,reward=injected.reward_accounting,
        active=injected._active,pending=injected._pending))==before
    assert len(records)==2 and all(not row['departed'] and not row['pending_decision'] for row in injected._vehicles.values())


def test_invalid_future_batch_is_rejected_before_any_registration():
    session=make_session(())
    connection=FakeConnection();session._connection=connection;session._time_ms=30000
    with pytest.raises(ValueError,match='at/before'):
        trial.register_future(session,[VehicleDemand('ok',0,0,40.),VehicleDemand('bad',0,0,30.)])
    assert not connection.routes and not connection.added
    with pytest.raises(ValueError,match='names collide'):
        trial.register_future(session,[VehicleDemand('same',0,0,40.),VehicleDemand('same',0,0,50.)])
    assert not connection.added


def test_controller_rejects_future_law_or_seed_fields():
    net=synthetic_network()
    controller=trial.GuardedController(net,fidelity.NextHopCandidates(net,traffic_model='meso'),40.)
    flow=np.zeros((4,8));flow[:,1]=1.
    controller.prepare(flow,300.)
    row=dict(vehicle_id='future_v0',current_edge=1,destination_edge=3)
    assert controller([row],flow,300.)[0].vehicle_id=='future_v0'
    with pytest.raises(ValueError,match='non-observation'):
        controller([dict(row,future_seed=810001)],flow,300.)
    assert controller.input_audit['schedule_or_rate_access'] is False


def test_event_protocol_transition_ignores_numeric_action_normalization():
    model=small_model().eval()
    events=pack_action_events([[(ActionEvent('a',0,0.,0,1,2),)]],window_starts=[[0.]])
    z=torch.zeros(1,3,64)
    action=torch.tensor([[0.,1.,0.]])
    noise=torch.zeros(1,model.dynamics.graph_prior.rank,64)
    with torch.inference_mode():
        a=model.transition(z,action/10,action_events=events,noise=noise,steps=2)
        b=model.transition(z,action/10000,action_events=events,noise=noise,steps=2)
    assert model.dynamics.event_protocol and torch.equal(a,b)


def test_model_rollout_keeps_raw_counts_and_own_events_separate(monkeypatch):
    net=synthetic_network()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(()))
            self.dynamics=SimpleNamespace(event_protocol=True,graph_prior=SimpleNamespace(rank=2))
            self.seen=[]
        def encode(self,flow):
            return torch.zeros(1,4,64)
        def decode(self,latent):
            decoded=torch.zeros(1,4,4)
            decoded[:,:,1]=2.  # Raw output must not be replaced by executor speed clipping.
            decoded[:,:,2]=.5
            return decoded
        def transition(self,latent,action,**kw):
            self.seen.append((action.clone(),kw['noise'].clone(),kw['action_events'].vehicle_ids))
            return latent
    class Arbiter:
        def __init__(self,controls,seed,uncertain_order):
            assert seed==29 and uncertain_order
            self.audit=[]
    def advance(states,**kw):
        assert len(states)==1 and states[0]['vehicle_id']=='past_v'
        assert kw['total_vehicle_count']==2  # Includes known arrived past, never unknown future population.
        states[0]['_city_queue_rank']=0
        decisions=kw['decide'](states,kw['frozen_flow'],kw['start_time'])
        event=ActionEvent('past_v',0,kw['start_time'],0,decisions[0].next_edge,3)
        return SimpleNamespace(action=np.zeros(4,dtype=np.float32),action_events=(event,),
            active_vehicle_seconds=30.,end_time=kw['start_time']+30.)
    monkeypatch.setattr(fidelity,'ChronologicalJunctionArbiter',Arbiter)
    monkeypatch.setattr(fidelity,'advance_city_imagined_window',advance)
    flow=np.zeros((4,8),dtype=np.float32);flow[:,1]=1.
    snapshot=SimpleNamespace(session=dict(_time_ms=300000,_flow_cache=(300000,flow),_vehicles={
        'past_v':dict(vehicle_id='past_v',current_edge=0,destination_edge=3,departed=True,arrived=False),
        'arrived_v':dict(vehicle_id='arrived_v',departed=True,arrived=True)}))
    model=Model().eval()
    before=fidelity.fingerprint(fidelity.external_rng_state())
    a=trial.model_rollout(model,snapshot,SimpleNamespace(network=net),None,draw_seed=17,executor_seed=29,steps=2)
    noise_a=[row[1] for row in model.seen]
    b=trial.model_rollout(model,snapshot,SimpleNamespace(network=net),None,draw_seed=17,executor_seed=29,steps=2)
    assert a==b and fidelity.fingerprint(fidelity.external_rng_state())==before
    assert len(a['frames'])==9 and len(a['windows'])==8
    assert all(torch.equal(x,y[1]) for x,y in zip(noise_a,model.seen[8:]))
    assert a['targets']['8']['vehicle_seconds']==240.
    assert a['targets']['8']['explicit_count_total']==1
    assert a['targets']['8']['raw_decoded_speed_mps']==(2*np.maximum(net.speed_limits_mps,.1)).tolist()
    assert a['targets']['8']['raw_decoded_count']==(.5*np.maximum(net.capacities,1.)).tolist()
    assert all(row['own_action_events'][0]['vehicle_id']=='past_v' for row in a['windows'])
    assert a['controller_input_audit']['forbidden_input_fields']==[]
    assert '_city_queue_rank' not in snapshot.session['_vehicles']['past_v']
