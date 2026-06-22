from scripts import analyze_dci_decoupled_communication as decoupled


def test_decoupled_relay_model_reuses_evidence_payload_once():
    evidence_report = {
        "relay_extra_comm_mb": 5.5,
        "normal_equation_summary_comm_mb": 0.4,
    }
    solve_report = {
        "actual_dci_comm_mb": 26.0,
        "relay_extra_comm_mb": 15.0,
        "normal_equation_summary_comm_mb": 0.4,
        "actual_linear_communication": {
            "linear_solve_total_estimated_mb": 11.0,
            "pcg_global_reduction_mb": 0.06,
        },
        "certificate_communication": {
            "residual_delta_consensus_mb": 0.001,
        },
    }

    estimate = decoupled.estimate_decoupled_communication(
        evidence_report,
        solve_report,
    )

    assert estimate["current_actual_mb"] == 26.0
    assert estimate["decoupled_relay_only_mb"] == 16.901
    assert estimate["relay_savings_mb"] == 9.099


def test_compact_continuation_model_keeps_only_global_reductions():
    evidence_report = {
        "relay_extra_comm_mb": 5.5,
        "normal_equation_summary_comm_mb": 0.4,
    }
    solve_report = {
        "actual_dci_comm_mb": 26.0,
        "relay_extra_comm_mb": 15.0,
        "normal_equation_summary_comm_mb": 0.4,
        "actual_linear_communication": {
            "linear_solve_total_estimated_mb": 11.0,
            "pcg_global_reduction_mb": 0.06,
        },
        "certificate_communication": {
            "residual_delta_consensus_mb": 0.001,
        },
    }

    estimate = decoupled.estimate_decoupled_communication(
        evidence_report,
        solve_report,
    )

    assert estimate["compact_continuation_mb"] == 5.961
    assert estimate["compact_savings_mb"] == 20.039
    assert estimate["decoupled_model"] == (
        "one_time_evidence_relay_plus_compact_solve_continuation"
    )
