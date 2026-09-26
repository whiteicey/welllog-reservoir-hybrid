import math

def multiplier(step,total=2560,warmup=128):
    step=min(max(int(step),0),total-1)
    if step<warmup:return .1+.9*step/(warmup-1)
    progress=(step-(warmup-1))/(total-warmup)
    return .01+.99*.5*(1+math.cos(math.pi*progress))

if __name__=='__main__':
    import torch
    parameter=torch.nn.Parameter(torch.tensor(0.))
    opt=torch.optim.AdamW([parameter],lr=.001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    values=[]
    for step in range(2560):
        values.append(opt.param_groups[0]['lr']);opt.step();sched.step()
    assert abs(values[0]-.0001)<1e-12 and abs(values[127]-.001)<1e-12 and abs(values[-1]-.00001)<1e-12
    assert all(a<=b for a,b in zip(values[:127],values[1:128]))
    assert all(a>=b for a,b in zip(values[127:-1],values[128:]))
    assert all(abs(value-.001*multiplier(i))<1e-12 for i,value in enumerate(values))
    print('PASS: actual optimizer learning rates; 128-step warmup and cosine end points and monotonic phases')
