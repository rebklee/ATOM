from atom.config import Config
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.spec_decode.eagle_proposer import EagleProposer


def _drafter_cls(config: Config) -> type[Drafter]:
    """Resolve the drafter and its configuration-only contracts together.

    Block-parallel DSpark vs serial EAGLE/MTP is the only flavor branch; every
    downstream call-site depends on the ``Drafter`` contract, not the concrete
    type.
    """
    if config.speculative_config.use_dspark():
        return DSparkProposer
    return EagleProposer


def build_drafter(config: Config, device, runner) -> Drafter:
    return _drafter_cls(config)(config, device, runner)


def build_pp_aux_relay(config: Config, target_model, drafter, device):
    """Use the drafter's aux contract on stages that do not load its weights."""
    from atom.spec_decode.pp_aux_relay import PPAuxRelay

    spec = _drafter_cls(config).config_aux_capture_spec(config)
    if spec is None:
        return None
    if drafter is not None and drafter._aux_capture_spec(target_model) != spec:
        raise NotImplementedError(
            "PP auxiliary capture must match the drafter's configuration contract"
        )
    return PPAuxRelay(config, target_model, drafter, device, spec)
