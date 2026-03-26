from __future__ import annotations

from src.methods.atta_style_active import ATTAStyleActiveAdapter
from src.methods.bitta_style_binary_feedback import BiTTAStyleBinaryFeedbackAdapter
from src.methods.come_zero_bit import ComeZeroBitAdapter
from src.methods.eatta_style_low_label import EATTAStyleLowLabelAdapter
from src.methods.fb_gate_tent import FBGateTentAdapter
from src.methods.frozen import FrozenAdapter
from src.methods.full_label import FullLabelActiveAdapter
from src.methods.self_gate_tent_zero_bit import SelfGateTentZeroBitAdapter
from src.methods.tent import TentAdapter
from src.methods.throttle_tent_zero_bit import ThrottleTentZeroBitAdapter


def build_adapter(method_name: str, model, device, config):
    registry = {
        "frozen": FrozenAdapter,
        "tent": TentAdapter,
        "fb_gate_tent": FBGateTentAdapter,
        "full_label_active": FullLabelActiveAdapter,
        "full_label_active_reference": FullLabelActiveAdapter,
        "atta_or_simatta_style_active": ATTAStyleActiveAdapter,
        "bitta_style_binary_feedback": BiTTAStyleBinaryFeedbackAdapter,
        "eatta_style_low_label": EATTAStyleLowLabelAdapter,
        "sar_or_come_zero_bit_stable": ComeZeroBitAdapter,
        "self_gate_tent_zero_bit": SelfGateTentZeroBitAdapter,
        "throttle_tent_zero_bit": ThrottleTentZeroBitAdapter,
    }
    if method_name not in registry:
        raise KeyError(f"Unknown method '{method_name}'.")
    return registry[method_name](model=model, device=device, config=config)
