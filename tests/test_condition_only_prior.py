import torch
import pytest
from dataclasses import replace
from urban_dreamer.models.world_model import Dynamics, independent_transition_fm_loss


def model(usage='condition_only'):
    torch.manual_seed(45)
    value = Dynamics(adjacency=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]),
        static_features=torch.zeros(3,5), low_frequency_rank=2, prior_filter='soft',
        prior_filter_strength=.1, prior_action_conditioned=False, prior_usage=usage).eval()
    value.transformer.activation_checkpointing = False
    with torch.no_grad():
        value.graph_prior.output.weight.normal_(std=.1)
        value.flow_head.weight.normal_(std=.01)
        if usage == 'condition_only':
            value.condition_residual.output.weight.normal_(std=.02)
    return value


def inputs():
    return torch.randn(2,1,3,64), torch.rand(2,1,3), torch.randn(2,1,2,64)


def test_background_condition_changes_velocity_not_source_reference():
    m=model();z,a,n=inputs();condition=m.encode_condition(a)
    start,b,c=m.prior_start(z,condition,n)
    torch.testing.assert_close(start,z+m.graph_prior.sample_noise(z,n))
    torch.testing.assert_close(c.reference_tokens,m.reference_proj(z))
    assert c.current_state is z  # retain complete, including high-frequency state
    torch.testing.assert_close(b,m.graph_prior(z,m.encode_condition(a+5)))
    t=torch.full((2,1),.5)
    v=m(a,packed_latents=start,flow_time=t,condition=c).latent_velocity
    changed=m(a,packed_latents=start,flow_time=t,condition=replace(c,background=c.background+1)).latent_velocity
    assert not torch.allclose(v,changed)
    with torch.no_grad(): m.graph_prior.output.bias.add_(1)
    start2,b2,c2=m.prior_start(z,condition,n)
    assert not torch.allclose(b,b2)
    torch.testing.assert_close(start,start2,rtol=0,atol=0)
    torch.testing.assert_close(c.reference_tokens,c2.reference_tokens,rtol=0,atol=0)


def test_fm_prior_gradient_isolation_and_live_auxiliary():
    m=model();z,a,n=inputs()
    loss,_,_=independent_transition_fm_loss(m,z_inputs=z,z_targets=z+.3,action=a,
        noise=n,flow_time=torch.full((2,1),.4),prior_weight=0,predict_endpoints=False)
    loss.backward()
    assert all(p.grad is None for p in m.graph_prior.parameters())
    assert m.condition_residual.hidden.weight.grad.abs().sum()>0
    m.zero_grad(set_to_none=True)
    loss,_,_=independent_transition_fm_loss(m,z_inputs=z,z_targets=z+.3,action=a,
        noise=n,flow_time=torch.full((2,1),.4),prior_weight=.1,predict_endpoints=False)
    loss.backward()
    assert m.graph_prior.output.weight.grad.abs().sum()>0


@pytest.mark.parametrize('solver',['euler','midpoint','picard'])
def test_disagreement_matches_transitions_and_picard_chunk_context(solver):
    m=model();z,a,_=inputs();noise=torch.randn(3,2,1,2,64)
    with torch.no_grad():
        expected=torch.stack([m.transition(z,a,noise=n,steps=2,solver=solver,
            picard_batch_chunk_size=1) for n in noise])
        actual,_,_=m.latent_disagreement(z,a,noise=noise,n_samples=3,steps=2,solver=solver,
            picard_batch_chunk_size=3)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)


def test_no_extra_rng_draw_for_disagreement():
    m=model();z,a,_=inputs()
    torch.manual_seed(5)
    noise=torch.randn(3,2,1,2,64)
    expected_rng=torch.random.get_rng_state()
    with torch.no_grad(): expected=m.latent_disagreement(z,a,noise=noise,n_samples=3,steps=1)[0]
    torch.manual_seed(5)
    with torch.no_grad(): actual=m.latent_disagreement(z,a,n_samples=3,steps=1)[0]
    assert torch.equal(torch.random.get_rng_state(),expected_rng)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_legacy_state_dict_and_initialization_unchanged():
    m=model('source_reference')
    assert not any('condition_residual' in k or 'condition_rms' in k for k in m.state_dict())
    torch.manual_seed(45)
    default=Dynamics(adjacency=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]),
        static_features=torch.zeros(3,5), low_frequency_rank=2, prior_filter='soft',
        prior_filter_strength=.1, prior_action_conditioned=False)
    # Test default constructor remains legacy, including keys and random initial values.
    torch.manual_seed(45)
    explicit=Dynamics(adjacency=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]),
        static_features=torch.zeros(3,5), low_frequency_rank=2, prior_filter='soft',
        prior_filter_strength=.1, prior_action_conditioned=False,prior_usage='source_reference')
    for k,v in default.state_dict().items(): torch.testing.assert_close(v,explicit.state_dict()[k],rtol=0,atol=0)


def test_condition_required_and_state_only_enforced():
    m=model();z,a,_=inputs()
    with pytest.raises(ValueError,match='current-state context'):
        m(a,packed_latents=z,flow_time=torch.zeros(2,1))
    with pytest.raises(ValueError,match='state-only'):
        Dynamics(adjacency=torch.eye(3),static_features=torch.zeros(3,5),prior_usage='condition_only')


def test_teacher_targets_never_enter_background_context():
    m=model();z,a,n=inputs();seen=[]
    def capture(_module,_args,kwargs):
        c=kwargs['condition']
        seen.append((c.current_state.detach().clone(),c.background.detach().clone()))
    handle=m.register_forward_pre_hook(capture,with_kwargs=True)
    for target in [z+1,z-100]:
        independent_transition_fm_loss(m,z_inputs=z,z_targets=target,action=a,noise=n,
            flow_time=torch.full((2,1),.4),prior_weight=0,predict_endpoints=False)
    handle.remove()
    for current,b in seen:
        torch.testing.assert_close(current,z)
        torch.testing.assert_close(b,m.graph_prior(z,m.encode_condition(a)))
    torch.testing.assert_close(seen[0][1],seen[1][1],atol=0,rtol=0)


def test_native_head_matches_direct_condition_formula_with_persistent_scales():
    m=model();z,a,n=inputs();start,_,c=m.prior_start(z,m.encode_condition(a),n)
    with torch.no_grad():
        m.condition_rms_z.fill_(1.1)
        m.condition_rms_background.fill_(.2)
        m.condition_rms_relative.fill_(.15)
    hidden=[]
    handle=m.flow_head.register_forward_pre_hook(lambda _m,args:hidden.append(args[0]))
    t=torch.full((2,1),.3)
    actual=m(a,packed_latents=start,flow_time=t,condition=c).latent_velocity
    handle.remove()
    h=hidden[0]
    features=torch.cat((h,z/1.1,c.background/.2,(start-z)/.15,t[...,None,None].expand(2,1,3,1)),dim=-1)
    expected=m.flow_head(h)+m.condition_residual(features)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert m.state_dict()['condition_rms_background'].item()==pytest.approx(.2)
