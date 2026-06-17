from urcm.core.logic_gates import ProgramSynth

def test_program_synth_sum():
    code = ProgramSynth.synthesize({"goal":"sum two numbers"})
    ok = ProgramSynth.verify(code, [((2,3),5), ((0,0),0)])
    assert ok is True

def test_program_synth_fail_case():
    code = ProgramSynth.synthesize({"goal":"unknown"})
    ok = ProgramSynth.verify(code, [((2,3),5)])
    assert ok is False

def test_program_contract_counterexample():
    code = ProgramSynth.synthesize({"goal":"sum two numbers"})
    c = ProgramSynth.Contract(
        arg_types=["int","int"],
        precond=lambda args: True,
        postcond=lambda args, out: out == args[0] + args[1]
    )
    res = ProgramSynth.verify_with_contracts(code, [c], samples=10)
    assert res["ok"] is True
