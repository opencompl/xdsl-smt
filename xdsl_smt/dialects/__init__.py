from typing import Callable
from xdsl.dialects import get_all_dialects as xdsl_get_all_dialects
from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    all_dialects = xdsl_get_all_dialects()

    def get_abbv_dialect():
        from xdsl_smt.dialects.ab_bitvector_dialect import ABBitVectorDialect

        return ABBitVectorDialect

    def get_pdl_dialect():
        from xdsl.dialects.pdl import PDL
        from xdsl_smt.dialects.pdl_dataflow import PDLDataflowDialect

        return Dialect(
            "pdl",
            [*PDL.operations, *PDLDataflowDialect.operations],
            [*PDL.attributes, *PDLDataflowDialect.attributes],
        )

    def get_smt_dialect():
        from xdsl_smt.dialects.smt_array_dialect import SMTArray
        from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
        from xdsl_smt.dialects.smt_dialect import SMTDialect
        from xdsl_smt.dialects.smt_int_dialect import SMTIntDialect
        from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
        from xdsl_smt.dialects.smt_tensor_dialect import SMTTensorDialect

        dialects = [
            SMTArray,
            SMTBitVectorDialect,
            SMTDialect,
            SMTIntDialect,
            SMTUtilsDialect,
            SMTTensorDialect,
        ]
        return Dialect(
            "smt",
            [op for dialect in dialects for op in dialect.operations],
            [attr for dialect in dialects for attr in dialect.attributes],
        )

    def get_effect_dialect():
        from xdsl_smt.dialects.effects.effect import EffectDialect

        return EffectDialect

    def get_effect_ub_dialect():
        from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect

        return UBEffectDialect

    def get_memory_effect_dialect():
        from xdsl_smt.dialects.effects.memory_effect import MemoryEffectDialect

        return MemoryEffectDialect

    def get_memory_dialect():
        from xdsl_smt.dialects.memory_dialect import MemoryDialect

        return MemoryDialect

    def get_transfer_dialect():
        from xdsl_smt.dialects.transfer import Transfer

        return Transfer

    def get_tv_dialect():
        from xdsl_smt.dialects.tv_dialect import TVDialect

        return TVDialect

    def get_hoare_dialect():
        from xdsl_smt.dialects.hoare_dialect import Hoare

        return Hoare

    def get_synth_dialect():
        from xdsl_smt.dialects.synth_dialect import SynthDialect

        return SynthDialect

    def get_tensor_dialect():
        from xdsl_smt.dialects.smt_tensor_dialect import SMTTensorDialect

        return SMTTensorDialect

    all_dialects["abbv"] = get_abbv_dialect
    all_dialects["pdl"] = get_pdl_dialect
    all_dialects["smt"] = get_smt_dialect
    all_dialects["effect"] = get_effect_dialect
    all_dialects["ub_effect"] = get_effect_ub_dialect
    all_dialects["memory_effect"] = get_memory_effect_dialect
    all_dialects["memory"] = get_memory_dialect
    all_dialects["transfer"] = get_transfer_dialect
    all_dialects["tv"] = get_tv_dialect
    all_dialects["hoare"] = get_hoare_dialect
    all_dialects["synth"] = get_synth_dialect
    all_dialects["tensor"] = get_tensor_dialect

    return all_dialects
