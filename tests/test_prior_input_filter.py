import numpy as np
import pytest
import torch

from tools.evaluate_prior_input_filter import (
    filtered_design, identity_equivalence, identity_reference_contrasts, paired_contrasts, state_hidden, sums, summarize,
)
from tools.evaluate_pretrained_prior_ranks import solve_ridge
from urban_dreamer.models.world_model import ConditionEncoding, LowFrequencyGraphPrior


@pytest.mark.parametrize('family', ['soft','hard'])
@pytest.mark.parametrize('filter_input', [False,True])
def test_native_both_end_formula_and_action_independence(family,filter_input):
    torch.manual_seed(23)
    adjacency=torch.tensor([[0.,1.,0.,0.],[0.,0.,1.,0.],[1.,0.,0.,1.],[0.,1.,0.,0.]])
    prior=LowFrequencyGraphPrior(adjacency,rank=2,prior_filter=family,
        prior_action_conditioned=False,prior_filter_input=filter_input).double()
    with torch.no_grad():
        prior.output.weight.normal_();prior.output.bias.normal_()
    z=torch.randn(2,1,4,64,dtype=torch.float64)
    condition=ConditionEncoding(torch.randn(2,1,4,256,dtype=torch.float64))
    h=state_hidden(prior,z)
    expected=prior.project(prior.output(h))
    torch.testing.assert_close(prior(z,condition),expected)
    torch.testing.assert_close(prior(z,ConditionEncoding(condition.edge_tokens*71)),expected,rtol=0,atol=0)
    operator=prior.project(torch.eye(4,dtype=torch.float64)).detach().numpy()
    x,y=filtered_design(h[:,0].detach().numpy(),np.zeros((2,4,64)),operator)
    coefficients=np.concatenate((prior.output.weight.detach().numpy().T,prior.output.bias.detach().numpy()[None]))
    np.testing.assert_allclose(x@coefficients,expected[:,0].detach().numpy(),rtol=1e-6,atol=3e-6)
    assert coefficients.shape==(65,64) and coefficients.size==4160
    assert x.dtype==y.dtype==np.float64


def test_soft_keeps_high_frequency_and_input_changes_nonlinear_front():
    prior=LowFrequencyGraphPrior(torch.tensor([[0.,1.],[1.,0.]]),rank=1,
        prior_filter='soft',prior_action_conditioned=False).double()
    z=torch.zeros(1,2,64,dtype=torch.float64);z[0,0,0]=10;z[0,1,0]=-10
    high=prior.project(z)
    assert 0<high.norm()<z.norm()
    torch.testing.assert_close(high,z/21,rtol=1e-6,atol=1e-8)
    with torch.no_grad():
        prior.state_proj.weight.copy_(torch.eye(64));prior.state_proj.bias.zero_()
        prior.upstream_proj.weight.zero_();prior.downstream_proj.weight.zero_()
    output_hidden=state_hidden(prior,z)
    prior.filter_input=True
    input_hidden=state_hidden(prior,z)
    assert not torch.allclose(output_hidden,input_hidden)
    # High-frequency input produces low-frequency features through SiLU.
    assert abs(output_hidden[0,:,0].mean())>abs(input_hidden[0,:,0].mean())


def test_filtered_design_targets_bias_and_independent_ridge_sse():
    rng=np.random.default_rng(93)
    hidden,delta=rng.normal(size=(3,5,4)),rng.normal(size=(3,5,4))
    operator=np.diag([1.,.5,.3,.2,.1])
    x,y=filtered_design(hidden,delta,operator)
    np.testing.assert_allclose(x[...,-1],np.broadcast_to(operator@np.ones(5),(3,5)))
    np.testing.assert_allclose(y,operator@delta)
    flat=x.reshape(-1,5);target=y.reshape(-1,4)
    gram,cross=flat.T@flat,flat.T@target
    head,proof=solve_ridge(gram,cross,np.square(y).sum())
    direct=np.square(x@head-y).sum()
    assert proof['training_projected_sse']==pytest.approx(direct)
    metrics=sums(delta,x@head,y,operator)
    assert metrics['filtered_target_sse']==pytest.approx(direct)
    assert metrics['filtered_error_sse']==pytest.approx(np.square(operator@(delta-x@head)).sum())


def test_group_bootstrap_pairs_targets_and_aggregates_episodes():
    records=[]
    for group,episode in [('a',1),('a',2),('b',3)]:
        for arm,amount in [('output_only',.8),('input_output',.6),('zero_prior',1.)]:
            row=dict(target='raw_next',filter='soft_tau10',arm=arm,demand_group=group,episode=episode,windows=2,
                total_delta_energy=10.,filtered_delta_energy=5.,full_sse=10*amount,filtered_target_sse=5*amount,
                filtered_error_sse=5*amount,prediction_energy=1.,dot_product=1.)
            for metric in ('count','speed'):
                for suffix in ('','_occupied'):
                    row[metric+suffix+'_absolute_error']=amount*2;row[metric+suffix+'_elements']=2
            records.append(row)
    groups,rows=summarize(records,repetitions=100)
    assert len(groups)==6
    assert next(r for r in rows if r['arm']=='input_output')['windows']==6
    contrasts=paired_contrasts(groups,repetitions=100)
    assert all(r['input_output_minus_output_only']==pytest.approx(-.2) for r in contrasts)
    assert all(r['groups']==2 for r in contrasts)


def test_zero_and_invalid_inputs():
    zeros=np.zeros((2,3,4));x,y=filtered_design(zeros,zeros,np.eye(3))
    assert all(v==0 for v in sums(zeros,zeros,y,np.eye(3)).values())
    assert x.shape==(2,3,5)
    with pytest.raises(ValueError,match='finite'):
        filtered_design(np.full_like(zeros,np.nan),zeros,np.eye(3))


def test_identity_input_output_front_and_fitted_design_are_bit_identical():
    adjacency=torch.ones(4,4)-torch.eye(4)
    a=LowFrequencyGraphPrior(adjacency,rank=2,prior_filter='soft',prior_filter_strength=0.,prior_action_conditioned=False)
    b=LowFrequencyGraphPrior(adjacency,rank=2,prior_filter='soft',prior_filter_strength=0.,prior_action_conditioned=False,prior_filter_input=True)
    b.load_state_dict(a.state_dict())
    z=torch.randn(3,4,64)
    h=identity_equivalence(a,b,z)
    delta=np.ones((3,4,64));operator=a.soft_filter_operator.numpy()
    x,y=filtered_design(h.detach().numpy(),delta,operator)
    x2,y2=filtered_design(state_hidden(b,z).detach().numpy(),delta,operator)
    np.testing.assert_array_equal(x,x2);np.testing.assert_array_equal(y,y2)
    head=np.arange(65*64).reshape(65,64)/10000
    np.testing.assert_array_equal(x@head,x2@head)


def test_identity_reference_comparison_uses_common_full_target_only():
    metric=dict(total_delta_energy=10.,filtered_delta_energy=10.,full_sse=6.,filtered_target_sse=6.,
        filtered_error_sse=6.,prediction_energy=1.,dot_product=1.,windows=2)
    for m in ('count','speed'):
        for suffix in ('','_occupied'):
            metric[m+suffix+'_absolute_error']=2.;metric[m+suffix+'_elements']=2
    controls=[dict(metric,target='raw_next',filter='identity_tau0',arm='output_only',demand_group='a')]
    references=[dict(metric,target='raw_next',filter='soft_tau10',arm=arm,demand_group='a',
        filtered_delta_energy=1.,full_sse=error) for arm,error in [('output_only',8.),('zero_prior',10.)]]
    result=identity_reference_contrasts(controls,references)
    assert all(r['metric']!='filtered_normalized_sse' for r in result)
    sse={r['reference']:r['identity_minus_reference'] for r in result if r['metric']=='full_normalized_sse'}
    assert sse==pytest.approx({'soft_tau10_output_only':-.2,'zero_prior':-.4})
