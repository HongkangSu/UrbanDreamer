from pathlib import Path
p=Path('tools/train_direct_context_fm.py')
s=p.read_text().replace('from tools.train_nonlinear_fm_residual import NonlinearResidual','from tools.adapt_fm_output_head import capture_final_features\nfrom tools.train_nonlinear_fm_residual import NonlinearResidual')
s=s.replace('started=time.time();torch.set_num_threads(2);','started=time.time();torch.set_num_threads(args.threads);')
s=s.replace("'epoch0 candidate and8samefixedstates4groupsK4Euler100 at20,25,...120; independentbestvalES'","f'epoch0 candidate and8samefixedstates4groupsK4Euler{args.steps} at20,25,...120; independentbestvalES'")
s=s.replace("'eachpriorarm eligible vszero ifES+bothoccupiedCRPS CIupper<0 andtotalcountpoint<=0; ifany qualifies fixedfirst8teststatesC0 H4Euler100K8'","f'eachpriorarm eligible vszero ifES+bothoccupiedCRPS CIupper<0 andtotalcountpoint<=0; ifany qualifies fixedfirst8teststatesC0 H4Euler{args.steps}K8'")
s=s.replace("precision='FP32noAMP/compile CPU/BLAS2',","precision=f'FP32noAMP/compile CPU/BLAS{args.threads}',integration_steps=args.steps,screening=args.steps!=100,\n        initial_gate='same-backend original vs zero residual exact; archived bit identity additionally required onlyCUDA100',\n        cache_compatibility='firsttraininggroup/firstquarter bothfamilies H/Y allrows: rtol1e-4 atol1e-4 beforeSGD; failclosed',")
s=s.replace("device=torch.device(args.device);torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)\n    if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')\n    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)","device=torch.device(args.device)\n    if device.type=='cuda':\n        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)\n        if free<args.gpu_memory_fraction*total+256*1024**2:raise RuntimeError('insufficient GPU headroom')\n        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device)")
s=s.replace('steps=100','steps=args.steps')
s=s.replace("                    with direct_context(model.dynamics,heads[arm],weak,arm,dict(z=1.,b=1.,relative=1.)):","                    baseline=trajectory_bank(model,z,batch.action.to(device),batch.action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(1,),samples=8,chunk=4,steps=args.steps)[1]\n                    with direct_context(model.dynamics,heads[arm],weak,arm,dict(z=1.,b=1.,relative=1.)):")
s=s.replace("                    np.testing.assert_array_equal(draws,saved['hybrid_state_soft0.1' if FAMILY[arm]=='weak_soft' else 'zero_mean'])\n                    gates.append(dict(condition=item['condition_index'],arm=arm,zero_head_exact_old=True))","                    np.testing.assert_array_equal(draws,baseline)\n                    archive_exact=device.type=='cuda' and args.steps==100\n                    if archive_exact:np.testing.assert_array_equal(draws,saved['hybrid_state_soft0.1' if FAMILY[arm]=='weak_soft' else 'zero_mean'])\n                    gates.append(dict(condition=item['condition_index'],arm=arm,zero_head_exact_old=True,archived_bit_identity_checked=archive_exact,device=str(device),steps=args.steps))")
s=s.replace("                if stage=='train':\n                    for key,value", """                if stage=='train' and ordinal==0:
                    times,noise,_=stratified_draws(args.seed,stage,e,starts,0,native.rank);times=times.to(device);noise=noise.to(z);compatibility=[]
                    for family,prior in priors.items():
                        model.dynamics.graph_prior=prior;source,_,cached=model.dynamics.prior_start(z,condition,noise)
                        interpolated=(1-times[:,:,None,None])*source+times[:,:,None,None]*target
                        features,out=capture_final_features(model.dynamics.flow_head,lambda:model.dynamics(batch.action.to(device),packed_latents=interpolated,flow_time=times,condition=cached))
                        for key,value in [('x',features),('y',target-source-out.latent_velocity)]:
                            actual=value.cpu().numpy().reshape(-1,256 if key=='x' else 64);expected=caches[stage][family][key][:len(actual)]
                            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
                            compatibility.append(dict(family=family,field=key,rows=len(actual),max_absolute_error=float(np.max(np.abs(actual-expected))),rtol=1e-4,atol=1e-4))
                    model.dynamics.graph_prior=native;atomic_json(args.output/'parent_cache_compatibility.json',compatibility)
                if stage=='train':
                    for key,value""")
s=s.replace('max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2','max_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type==\'cuda\' else 0.')
s=s.replace("    p.add_argument('--device',default='cuda:0');", "    p.add_argument('--threads',type=int,default=2);p.add_argument('--steps',type=int,default=100);\n    p.add_argument('--device',default='cuda:0');")
p.write_text(s)
