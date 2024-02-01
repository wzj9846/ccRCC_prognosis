import torch
import torch.nn as nn

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()
        self.epsilon=1
    def forward(self, risk_scores, survival_time, survival_status):
        exp_risk_scores = torch.exp(risk_scores)
        risk_set = (survival_time.unsqueeze(1) <= survival_time).float()
        total_risk_scores = torch.sum(exp_risk_scores * risk_set, dim=1)
        log_cumulative_risk = torch.log(torch.cumsum(total_risk_scores, dim=0))
        censor_mask = (survival_status == 0).float()
        loss = -torch.sum(risk_scores.squeeze() * censor_mask - log_cumulative_risk  * censor_mask)/(torch.sum(survival_status)+self.epsilon)
        
        return loss
    
