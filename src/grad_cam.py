import torch
def referral_grad_cam(model,modality,image):
    model.eval();features=[];branch=getattr(model,modality.lower())
    hook=branch.encoder.features[-1].register_forward_hook(lambda module,args,output:features.append(output))
    try:
        with torch.enable_grad():
            prediction=model.forward_modality(modality,image.detach().requires_grad_(True))
            logits=prediction['grade_logits']
            score=torch.logsumexp(logits[:,2:],dim=1)-torch.logsumexp(logits[:,:2],dim=1)
            gradient=torch.autograd.grad(score.sum(),features[0])[0]
            alpha=gradient.mean(dim=(-2,-1),keepdim=True)
            cam=torch.relu((alpha*features[0]).sum(dim=1,keepdim=True))
        return cam.detach(),score.detach()
    finally:hook.remove()
