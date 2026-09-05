import torch
import pytest
from dl_moe.experiments.dl_moe_01.evaluate import classification_metrics, evaluate_checkpoint
from dl_moe.model import DLMoE,DLMoEConfig


def test_balanced_accuracy_is_mean_class_recall():
    r=classification_metrics(torch.ones(10),torch.tensor([1.]*9+[0.]))
    assert r['accuracy']==pytest.approx(.9)
    assert r['balanced_accuracy']==.5


def test_chunking_preserves_metrics_and_interventions_preserve_weights():
    torch.manual_seed(5)
    m=DLMoE(DLMoEConfig(model_dim=8,expert_hidden_dim=12,experts_per_population=2,comm_dim=2))
    before={k:v.clone() for k,v in m.state_dict().items()}
    a=evaluate_checkpoint(m,40,123,40)
    b=evaluate_checkpoint(m,40,123,7)
    for task in ('A','B'):
        assert a['behavior'][task]['balanced_accuracy']==b['behavior'][task]['balanced_accuracy']
    assert a['expert']==b['expert']
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    assert len(a['expert']['renormalized'])==4
    assert len(a['communication_knockout'])==3
