"""Summarize audited frozen-AC counterfactual runs without retraining."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.counterfactual_statistics import cluster_paired_summary, paired_summary

COMPONENTS = ("shortest_ff_seconds", "route_excess_seconds", "residual_delay_seconds")
FIELDS = ("travel_seconds",) + COMPONENTS


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def state_identity(metrics):
    state = metrics.get("state") or {}
    keys = ("time", "sumo_sha256", "python_sha256", "decision_prefix_sha256")
    if not all(key in state for key in keys):
        raise ValueError(f"Missing state audit in {metrics.get('label')}")
    return {key: state[key] for key in keys}


@lru_cache(maxsize=8)
def load_routes(path):
    with gzip.open(path, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    result = {row["vehicle_id"]: row for row in records}
    if len(result) != len(records):
        raise ValueError("Duplicate vehicle IDs in route records")
    return result


def route_records(seed_dir, metrics):
    return load_routes(seed_dir / metrics["routes"])


def compare_pair(pair, plan, seed_dir, kind):
    before, after = (pair["a1"], pair["a2"]) if kind == "single" else (pair["baseline"], pair["treatment"])
    if state_identity(before) != state_identity(after):
        raise ValueError("Paired branches do not share the same audited decision state")
    if before["demand_sha256"] != after["demand_sha256"] or before["demand_sha256"] != plan["demand_sha256"]:
        raise ValueError("Paired exogenous demand differs")
    if before["vehicle_count"] != after["vehicle_count"]:
        raise ValueError("Paired demand population differs")
    complete = before["unfinished"] == after["unfinished"] == 0
    before_records, after_records = route_records(seed_dir, before), route_records(seed_dir, after)
    common = sorted(before_records.keys() & after_records.keys())
    if complete and (before_records.keys() != after_records.keys() or len(common) != before["vehicle_count"]):
        raise ValueError("Full-completion route populations differ from the shared demand")
    if any(not np.isclose(before_records[vid]["shortest_ff_seconds"], after_records[vid]["shortest_ff_seconds"], atol=1e-7, rtol=0) for vid in common):
        raise ValueError("Per-vehicle shortest free-flow cost differs on the matched population")
    means = {}
    if complete:
        for name, metrics in (("baseline", before), ("treatment", after)):
            means[name] = {field: float(metrics["decomposition"][field]) for field in FIELDS}
            if not np.isclose(means[name]["travel_seconds"], metrics["mean_travel_seconds"], atol=1e-7, rtol=0):
                raise ValueError("Decomposition and trip mean disagree")
        denominator = int(before["vehicle_count"])
        population = "all_realized_demand"
    else:
        if not common:
            raise ValueError("No common completed trips: a travel-time comparison is unavailable")
        denominator = len(common)
        for name, records in (("baseline", before_records), ("treatment", after_records)):
            means[name] = {field: float(np.mean([records[vid][field] for vid in common])) for field in FIELDS}
        population = "pairwise_common_completers_only"
    shortest_difference = means["baseline"]["shortest_ff_seconds"] - means["treatment"]["shortest_ff_seconds"]
    if not np.isclose(shortest_difference, 0, atol=1e-7, rtol=0):
        raise ValueError("Shortest free-flow cost differs on the matched population")
    differences = {field: means["baseline"][field] - means["treatment"][field] for field in FIELDS}
    if not np.isclose(differences["travel_seconds"], sum(differences[field] for field in COMPONENTS), atol=1e-7, rtol=0):
        raise ValueError("Paired decomposition identity failed")
    row = dict(demand_seed=int(pair["demand_seed"]), repeat=int(pair["repeat"]),
               future_seed=int(pair.get("future_seed", before["future_seed"])),
               difference=differences["travel_seconds"], population=population,
               denominator=denominator, vehicle_count=int(before["vehicle_count"]),
               baseline_unfinished=int(before["unfinished"]), treatment_unfinished=int(after["unfinished"]),
               baseline_completion_rate=float(before["completion_rate"]), treatment_completion_rate=float(after["completion_rate"]),
               all_population_complete=complete, shortest_ff_equal=True,
               state_sha256=state_identity(before)["sumo_sha256"],
               python_state_sha256=state_identity(before)["python_sha256"],
               decision_prefix_sha256=state_identity(before)["decision_prefix_sha256"],
               demand_sha256=before["demand_sha256"],
               reward_improvement_recorded=float(pair["reward_difference"]))
    departures = [before_records[vid]["actual_departure_time"] - after_records[vid]["actual_departure_time"] for vid in common]
    row["actual_departure_changed_vehicle_count"] = sum(abs(delta) > 1e-9 for delta in departures)
    row["mean_actual_departure_time_difference_seconds"] = float(np.mean(departures))
    # The same realized vehicle schedules cancel from baseline-minus-treatment.
    row["derived_release_to_arrival_improvement_seconds"] = differences["travel_seconds"] + row["mean_actual_departure_time_difference_seconds"]
    if kind == "single":
        row["delta_Q_user_R_a1_minus_R_a2"] = -float(pair["reward_difference"])
    for field in FIELDS:
        row[f"baseline_{field}"] = means["baseline"][field]
        row[f"treatment_{field}"] = means["treatment"][field]
        row[f"difference_{field}"] = differences[field]
    if complete and not np.isclose(row["reward_improvement_recorded"], row["difference"] / 60, atol=1e-8, rtol=0):
        raise ValueError("Reward improvement is not trip-time improvement / 60")
    if kind == "single":
        row.update(state_time=float(plan["target"]["time"]), vehicle_id=plan["target"]["vehicle_id"],
                   a1=int(plan["target"]["a1"]), a2=int(plan["target"]["a2"]))
        if len(common) != denominator:
            raise ValueError("Route records disagree with the matched trip denominator")
        target_id = plan["target"]["vehicle_id"]
        row["target_vehicle_common_completed"] = target_id in common
        for field in FIELDS:
            direct = (before_records[target_id][field] - after_records[target_id][field]) if target_id in common else None
            total = sum(before_records[vid][field] - after_records[vid][field] for vid in common)
            other = total - (direct if direct is not None else 0.)
            row[f"target_vehicle_difference_{field}"] = direct
            row[f"other_vehicles_total_difference_{field}"] = other
            row[f"target_contribution_to_global_mean_{field}"] = direct / denominator if direct is not None else None
            row[f"other_contribution_to_global_mean_{field}"] = other / denominator
            if not np.isclose(total / denominator, differences[field], atol=1e-7, rtol=0):
                raise ValueError("Direct and spillover effects do not reconstruct the paired mean")
    else:
        pilot_n = len(set(plan["target_ids"]))
        assigned, actual = int(after["assigned_target_count"]), int(after["intervention_count"])
        row.update(fraction=float(pair["fraction"]), pilot_eligible_vehicle_count=pilot_n,
                   assigned_target_count=assigned, intervention_count=actual,
                   live_eligible_vehicle_count=int(after["eligible_vehicle_count"]),
                   assigned_fraction_of_pilot=assigned / pilot_n if pilot_n else None,
                   intervention_fraction_of_pilot=actual / pilot_n if pilot_n else None,
                   intervention_fraction_of_all_demand=actual / before["vehicle_count"] if before["vehicle_count"] else None,
                   selected_target_uptake=actual / assigned if assigned else None)
        edge = plan["bottleneck_edge_id"]
        for name, metrics in (("baseline", before), ("treatment", after)):
            routes = route_records(seed_dir, metrics)
            visits = [record["external_path"].count(edge) for record in routes.values()]
            row[f"{name}_bottleneck_total_visits_completed"] = sum(visits)
            row[f"{name}_bottleneck_distinct_vehicles_completed"] = sum(count > 0 for count in visits)
        for metric in ("total_visits_completed", "distinct_vehicles_completed"):
            row[f"difference_bottleneck_{metric}"] = row[f"baseline_bottleneck_{metric}"] - row[f"treatment_bottleneck_{metric}"]
    return row


def component_summary(rows):
    return {field: cluster_paired_summary(rows, value_key=f"difference_{field}") for field in FIELDS}


def summarize_rows(rows, pilot_seed):
    holdout = [row for row in rows if row["demand_seed"] != pilot_seed]
    result = dict(all_demands=cluster_paired_summary(rows), holdout_excluding_pilot=cluster_paired_summary(holdout),
                component_improvements=component_summary(rows), holdout_component_improvements=component_summary(holdout),
                all_population_inference_permitted=all(row["all_population_complete"] for row in rows),
                denominator_range=[min(row["denominator"] for row in rows), max(row["denominator"] for row in rows)] if rows else None)
    result["departure_time_shift"] = cluster_paired_summary(rows, value_key="mean_actual_departure_time_difference_seconds")
    result["derived_release_to_arrival_improvement"] = cluster_paired_summary(rows, value_key="derived_release_to_arrival_improvement_seconds")
    result["holdout_derived_release_to_arrival_improvement"] = cluster_paired_summary(holdout, value_key="derived_release_to_arrival_improvement_seconds")
    if rows and "delta_Q_user_R_a1_minus_R_a2" in rows[0]:
        result["delta_Q_user_R_a1_minus_R_a2"] = cluster_paired_summary(rows, value_key="delta_Q_user_R_a1_minus_R_a2")
        result["holdout_delta_Q_user_R_a1_minus_R_a2"] = cluster_paired_summary(holdout, value_key="delta_Q_user_R_a1_minus_R_a2")
    if rows and "fraction" in rows[0]:
        result["bottleneck_completed_route_changes"] = {
            metric: cluster_paired_summary(rows, value_key=f"difference_bottleneck_{metric}")
            for metric in ("total_visits_completed", "distinct_vehicles_completed")}
        result["actual_interventions"] = {
            key: dict(minimum=min(row[key] for row in rows), maximum=max(row[key] for row in rows),
                      mean=float(np.mean([row[key] for row in rows])))
            for key in ("pilot_eligible_vehicle_count", "assigned_target_count", "intervention_count",
                        "intervention_fraction_of_pilot", "intervention_fraction_of_all_demand")}
    return result


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted(value, digits=3):
    return "不可估计" if value is None else f"{value:.{digits}f}"


def interval(summary):
    return f"[{formatted(summary['ci_low'])}, {formatted(summary['ci_high'])}]"


def make_plots(output, single, joint, states, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    ordered = sorted(states, key=lambda row: row["demand_seed"])
    for index, state in enumerate(ordered):
        effect = state["paired_effect"]
        mean = effect["mean"]
        axes[0].plot(index, mean, "o", color="tab:blue")
        if effect["ci_low"] is not None:
            axes[0].vlines(index, effect["ci_low"], effect["ci_high"], color="tab:blue")
    axes[0].set_xticks(range(len(ordered)), [str(row["demand_seed"]) for row in ordered], rotation=65)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(xlabel="Demand / decision-state seed", ylabel="T(a1) - T(a2) (s)",
                title="Single-vehicle paired effects\n95% continuation bootstrap intervals")

    doses = sorted({row["fraction"] for row in joint})
    for index, dose in enumerate(doses):
        effect = summary["joint"][str(dose)]["all_demands"]
        holdout = summary["joint"][str(dose)]["holdout_excluding_pilot"]
        for offset, item, color, label in ((-.08, effect, "tab:blue", "All demands"),
                                          (.08, holdout, "tab:orange", "Exclude pilot")):
            if item["mean"] is None:
                continue
            axes[1].plot(index + offset, item["mean"], "o", color=color, label=label if index == 0 else None)
            if item["ci_low"] is not None:
                axes[1].vlines(index + offset, item["ci_low"], item["ci_high"], color=color)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(range(len(doses)), [f"{dose:.0%}" for dose in doses])
    axes[1].set(xlabel="Assigned fraction of pilot-eligible vehicles", ylabel="T(0%) - T(dose) (s)",
                title="Joint intervention paired effects\n95% demand-cluster bootstrap intervals")
    axes[1].legend(fontsize=8)

    bars = []
    # Each seed is equally weighted; continuation repeats are averaged first.
    baseline_rows = list({(row["demand_seed"], row["repeat"]): row for row in joint}.values())
    if baseline_rows:
        bars.append(("0%", baseline_rows, "baseline"))
    bars.extend((f"{dose:.0%}", [row for row in joint if row["fraction"] == dose], "treatment") for dose in doses)
    colors = ("#94b8d7", "#e9b567", "#bf7777")
    labels = ("Shortest free-flow", "Route excess", "Residual delay")
    bottom = np.zeros(len(bars))
    for component, color, label in zip(COMPONENTS, colors, labels):
        values = []
        for _, rows, arm in bars:
            seeds = sorted({row["demand_seed"] for row in rows})
            values.append(float(np.mean([np.mean([row[f"{arm}_{component}"] for row in rows if row["demand_seed"] == seed]) for seed in seeds])))
        axes[2].bar(range(len(bars)), values, bottom=bottom, color=color, label=label)
        bottom += np.asarray(values)
    axes[2].set_xticks(range(len(bars)), [bar[0] for bar in bars])
    axes[2].set(xlabel="Assigned fraction of pilot-eligible vehicles", ylabel="Mean trip components (s)",
                title="Absolute trip decomposition\nEqual-weight demand means")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    complete = summary["completion"]["all_branches_complete"]
    if not complete:
        fig.suptitle("CENSORED: common-completer effects only; component populations vary by pair", color="darkred")
    else:
        protocol = summary["provenance"]["protocol"]
        fig.suptitle(f"Frozen AC baseline (episode {protocol['checkpoint_episode']}, T={protocol['temperature']}): exploratory counterfactual diagnostics")
    fig.savefig(output / "counterfactual_effects.png", dpi=200)
    fig.savefig(output / "counterfactual_effects.pdf")
    plt.close(fig)


def make_report(output, protocol, completed, plans, single, joint, states, summary):
    effect = summary["single"]["all_demands"]
    holdout = summary["single"]["holdout_excluding_pilot"]
    complete = summary["completion"]["all_branches_complete"]
    verification = read_json(output / "verification.json") if (output / "verification.json").exists() else {}
    verified = verification.get("status") == "pass"
    lines = ["# 冻结 AC baseline 的下一跳与联合分流反事实实验", "",
             "## Material Passport", "",
             "- Origin Skill: academic-research-suite/experiment-agent",
             "- Origin Mode: run",
             f"- Origin Date: {protocol['started_at']}",
             f"- Verification Status: {'VERIFIED' if verified else 'ANALYZED'}",
             "- Version Label: exp_result_v1", "",
             f"本实验使用已有 online AC 第 {protocol['checkpoint_episode']} 轮 Actor，采样温度 T={protocol['temperature']}，不重新训练（training_updates={protocol['training_updates']}）。",
             f"完成 {len(plans)} 个需求种子、{len(single)} 对单车分支、{len(joint)} 对联合分支；单车总体均值 ΔT={formatted(effect['mean'])} 秒，需求聚类描述性 95% 区间 {interval(effect)}。排除瓶颈发现用的首个 pilot 需求后，ΔT={formatted(holdout['mean'])} 秒，区间 {interval(holdout)}，独立需求数 {holdout['n']}。",
             "正值表示替代动作/分流降低全局平均完整行程时间；负值表示增加。单车 ΔT=T(a₁)−T(a₂)，联合 ΔT=T(0%)−T(分流比例)。全完成时奖励改善等于 ΔT/60，所以奖励与行程时间给出同样的排序。", "",
             "用户原式 ΔQ=R(a₁)−R(a₂) 与上述改善方向相反：全完成时 ΔQ=−ΔT/60。CSV 与 JSON 同时保留这一原始 ΔQ，避免把正改善误解为正 ΔQ。",
             "",
             "时间口径：这里的 Ttrip 和既有 global_reward 是 SUMO 实际插入/出发至到达的时间，不包含车辆在计划释放后、实际插入路网之前的网外等待。因此主结果不能直接称为计划释放至到达的完整系统耗时。配对路线另外记录实际出发时刻：相同外生释放计划下，计划释放至到达改善量=上述 ΔT+平均(实际出发时刻 baseline−treatment)。CSV/JSON 保留这一导出量及出发时刻改变的车辆数，用于检查插入等待是否改变。",
             "",
             "## 实验控制与适用范围", "",
             "每对分支从完整相同需求的前缀重新执行，到干预点比较含 RNG 的 SUMO 状态哈希、Python 控制状态哈希、决策前缀哈希；汇总再次核查相等性。未来外生车辆需求完全相同；Actor 冻结，其他车辆继续根据各分支真实交通状态作决策。随机数按 policy_seed、vehicle_id、decision_ordinal 的 SHA256 键产生，并通过分类分布逆 CDF 采样，避免批次车辆排序改变随机数配对。不同 continuation seed 估计同一 S 的后续策略随机性。",
             "该键控随机流与以前温度扫描中的批次 multinomial 不同，因此不能把本实验的 0% 数值直接当作旧评估的精确复现。干预只指定一次合法下一跳，后续 Actor 可以重新选路；它不强制整段绕行。",
             f"瓶颈为 {protocol.get('bottleneck_edge_id', protocol.get('bottleneck'))}，由首个需求 {summary['pilot_seed']} 中有合法绕行机会的道路按拥堵负担选取，单车状态也是各需求 pilot 的首个符合条件事件。这种有意选取的状态不是随机路由状态样本。固定瓶颈后，排除首个需求的 holdout 是独立需求诊断，但每个需求仍用自己的 discovery 选择目标状态和车辆集合，不能视作预先盲选所有状态的验证。",
             "默认设计共 10 个需求种子：70042–70044 对应此前温度实验的 3 个种子，70045–70051 是 7 个新增种子；实际本次种子为 " + ", ".join(str(seed) for seed in sorted(plans)) + "。", "",
             "## 单车反事实", "",
             "每行对应一个真实决策状态；n 是不同后续策略随机种子的配对数。SD 是配对 ΔT 的样本标准差，SNR=|均值|/SD，只刻画效应相对 continuation 噪声，不是显著性检验。", "",
             "|需求|时间(s)|车辆|a₁→a₂|n|均值 ΔT(s)|SD(s)|SNR|95% 描述区间|正/负/零|",
             "|---|---:|---|---|---:|---:|---:|---:|---|---|"]
    for state in states:
        item = state["paired_effect"]
        target = state["target"]
        lines.append(f"|{state['demand_seed']}|{target['time']:.0f}|{target['vehicle_id']}|{target['a1']}→{target['a2']}|{item['n']}|{formatted(item['mean'])}|{formatted(item['sd'])}|{formatted(item['signal_to_noise'])}|{interval(item)}|{item['positive_count']}/{item['negative_count']}/{item['zero_count']}|")
    distribution = summary["single_state_effect_distribution"]
    lines += ["", f"状态效应均值分布：最小 {formatted(distribution['minimum'])}、中位数 {formatted(distribution['median'])}、最大 {formatted(distribution['maximum'])} 秒。描述区间全在零上方的状态 {distribution['positive_interval_count']}/{distribution['n_states']}，全在零下方 {distribution['negative_interval_count']}/{distribution['n_states']}；未校正多状态比较，因此这些比例只描述观测区间，不宣称显著性。", "",
              "共同随机数给出的配对 SD 是在选定 S 下、冻结策略后续随机性对应的差值噪声；它不等于训练中跨状态、跨需求、随 Actor 更新产生的原始 return 方差。",
              "", "|需求|目标车辆自身 ΔT(s)|其他车辆 Δ总时间(s)|目标对全局均值贡献(s)|其他对全局均值贡献(s)|用户原式 ΔQ|",
              "|---|---:|---:|---:|---:|---:|"]
    for state in states:
        items = state["direct_and_spillover"]
        lines.append(f"|{state['demand_seed']}|{formatted(items['target_vehicle_difference_travel_seconds']['mean'])}|{formatted(items['other_vehicles_total_difference_travel_seconds']['mean'])}|{formatted(items['target_contribution_to_global_mean_travel_seconds']['mean'])}|{formatted(items['other_contribution_to_global_mean_travel_seconds']['mean'])}|{formatted(state['delta_Q_user_R_a1_minus_R_a2']['mean'], 5)}|")
    lines += ["", "由逐车完整路线记录计算直接效应与其他车辆效应；二者总时间差相加再除以配对分母，已核查重构全局 ΔT。其他车辆总时间差是 spillover 描述，不能直接等同于纯拥堵机制；其路线也会随真实状态变化。", "",
              "## 联合分流", "",
              "5%、10%、20% 是对 discovery 中瓶颈附近有合法替代下一跳的目标车辆集合做嵌套哈希抽样的指定比例，不是所有需求车辆的分流比例。被指定车辆只有在后续真实状态仍出现合格决策时才会干预，故指定数、实际干预数和全需求比例必须分开。", "",
              "|指定比例|需求数|配对数|ΔT 均值(s)|SD 跨需求(s)|95% 描述区间|排除 pilot 均值(s)|排除 pilot 区间|",
              "|---|---:|---:|---:|---:|---|---:|---|"]
    for dose, item in summary["joint"].items():
        all_effect, held = item["all_demands"], item["holdout_excluding_pilot"]
        lines.append(f"|{float(dose):.0%}|{all_effect['n']}|{all_effect['n_pairs']}|{formatted(all_effect['mean'])}|{formatted(all_effect['sd'])}|{interval(all_effect)}|{formatted(held['mean'])}|{interval(held)}|")
    lines += ["", "|指定比例|pilot 合格车辆数范围|指定目标数范围|实际干预数范围|指定目标执行率均值|实际干预/全需求均值|Δ绕行(s)|Δ残余延误(s)|",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for dose in sorted({row["fraction"] for row in joint}):
        rows = [row for row in joint if row["fraction"] == dose]
        span = lambda key: f"{min(row[key] for row in rows)}–{max(row[key] for row in rows)}"
        uptake = [row["selected_target_uptake"] for row in rows if row["selected_target_uptake"] is not None]
        population_fraction = [row["intervention_fraction_of_all_demand"] for row in rows if row["intervention_fraction_of_all_demand"] is not None]
        components = summary["joint"][str(dose)]["component_improvements"]
        lines.append(f"|{dose:.0%}|{span('pilot_eligible_vehicle_count')}|{span('assigned_target_count')}|{span('intervention_count')}|{np.mean(uptake):.1%}|{np.mean(population_fraction):.2%}|{formatted(components['route_excess_seconds']['mean'])}|{formatted(components['residual_delay_seconds']['mean'])}|")
    lines += ["", "|指定比例|减少的瓶颈完整路线访问次数|减少的曾经过瓶颈车辆数|",
              "|---|---:|---:|"]
    for dose, item in summary["joint"].items():
        changes = item["bottleneck_completed_route_changes"]
        lines.append(f"|{float(dose):.0%}|{formatted(changes['total_visits_completed']['mean'])}|{formatted(changes['distinct_vehicles_completed']['mean'])}|")
    lines += ["", "瓶颈访问统计按实际完成路线 external_path 计数，重复进入同一道路计作多次访问；正数表示 treatment 使用瓶颈更少。它检验单次下一跳干预是否最终改变瓶颈使用，后续 Actor 可能返回瓶颈。若未全完成，这些计数只描述已完成路线，不覆盖未完成车辆。"]
    lines += ["", "成本分解采用 T=最短自由流成本+实际路线自由流绕行成本+残余延误。相同完成车辆集合的最短自由流成本已核查相等。表中分解差值同样是 baseline−treatment：Δ绕行为负代表增加绕行；Δ残余延误为正代表减少延误。残余延误还包含 SUMO 离散时步、路口/排队及其他相对自由流的差异，不能全部等同于纯拥堵。", "",
              "## 完成率、估计边界与产物", ""]
    if complete:
        lines.append(f"所有纳入分支 unfinished=0，完整行程比较覆盖每对相同的全部需求车辆。配对分母范围为 {summary['completion']['denominator_range']}。")
    else:
        lines.append("存在 unfinished>0 的分支：禁止把完成者均值当成全需求平均行程时间。此处效应仅在每对都完成的车辆交集上重新计算；交集随分支、剂量变化，存在选择偏差。不能据此断言全系统行程改善，原始奖励改善只作为记录保留。")
    changed_pairs = sum(row["actual_departure_changed_vehicle_count"] > 0 for row in single + joint)
    lines.append(f"实际出发时刻发生改变的配对数为 {changed_pairs}/{len(single) + len(joint)}；单车平均出发时刻差为 {formatted(summary['single']['departure_time_shift']['mean'], 6)} 秒。" +
                 ("所有已比较车辆的实际出发时刻相同，故这些配对的网外等待差为零，主 ΔT 与计划释放至到达耗时差一致。" if changed_pairs == 0 else
                  "存在插入时刻变化，主 ΔT 与计划释放至到达耗时差需分开解读；各剂量导出量见 summary.json。"))
    lines.append(f"所有配对的平均出发时刻差范围为 {summary['departure_time_audit']['mean_departure_difference_range_seconds']} 秒，最大绝对校正 {formatted(summary['departure_time_audit']['maximum_absolute_mean_departure_difference_seconds'], 6)} 秒。各联合剂量的需求聚类平均校正（baseline−treatment）为 " +
                 "、".join(f"{float(dose):.0%}: {formatted(item['departure_time_shift']['mean'], 6)} 秒" for dose, item in summary['joint'].items()) + "。")
    lines += ["每个需求先平均其配对差，再等权聚合需求，整体区间重采样独立需求；没有把同状态 continuation 当成额外独立需求。每个状态区间重采样不同 continuation。20,000 次 percentile bootstrap 只作描述，小样本覆盖可能不足，零方差区间不能外推为确定性证据；三种联合剂量和多个状态均未作多重比较校正，因此不报告校正后的显著性结论。",
              "判定当前路由信号应同时查看效应方向、每个状态的 SD/SNR、描述区间、跨需求稳定性、实际干预规模及绕行/延误分解。本实验只回答选定瓶颈与状态上的动作敏感性，不证明其他状态普遍可学习或重新训练必然改善。", "",
              "产物：`single_pairs.csv`、`joint_pairs.csv` 保留每个配对的分母、完成率、状态/需求哈希与成本分解；`summary.json` 包含所有需求及排除 pilot 的汇总、每状态分布和 provenance；`counterfactual_effects.png` 与 `.pdf` 是可导出的三面板科学图。", "",
              f"Actor SHA256：`{protocol['actor_sha256']}`。completed.json 的 actor_unchanged={completed.get('actor_unchanged')}。源码：`{protocol['source']}`。运行开始于 {protocol['started_at']}，完成于 {completed.get('completed_at', '尚未全部完成')}。"]
    python = shlex.quote(sys.executable)
    destination = shlex.quote(str(output))
    seed_flags = " ".join(str(seed) for seed in protocol["demand_seeds"])
    lines += ["", "## 复现命令", "", "从项目根目录执行。重新运行须将输出路径换成一个不存在的新目录，runner 会拒绝覆盖现有实验。", "", "```bash",
              f"{python} tools/run_ac_counterfactual.py --output {destination} --seeds {seed_flags} --temperature {protocol['temperature']} --repeats {len(protocol['future_policy_seeds'])} --workers {protocol['workers']}",
              f"{python} tools/summarize_ac_counterfactual.py --output {destination}",
              f"{python} tools/verify_ac_counterfactual.py --output {destination}", "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ac_counterfactual_20261010")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "protocol.json").exists():
        parser.exit(1, "No experiment protocol or complete results found.\n")
    protocol = read_json(output / "protocol.json")
    if not (output / "completed.json").exists():
        parser.exit(1, "Experiment is not complete: completed.json missing; no summary written.\n")
    completed = read_json(output / "completed.json")
    if completed.get("actor_unchanged") is not True or completed.get("training_updates") != 0:
        parser.exit(1, "Frozen-actor audit failed; no summary written.\n")
    plans, single, joint, states = {}, [], [], []
    expected_single = len(protocol["future_policy_seeds"])
    expected_joint = 3 * min(expected_single, 3)
    omitted = []
    for seed in protocol["demand_seeds"]:
        seed_dir = output / f"seed_{seed}"
        paths = [seed_dir / name for name in ("discovery_plan.json", "single_pairs.json", "joint_pairs.json")]
        if not all(path.exists() for path in paths):
            omitted.append(seed)
            continue
        plan, singles, joints = map(read_json, paths)
        if len(singles) != expected_single or len(joints) != expected_joint:
            omitted.append(seed)
            continue
        plans[seed] = plan
        seed_single = [compare_pair(pair, plan, seed_dir, "single") for pair in singles]
        seed_joint = [compare_pair(pair, plan, seed_dir, "joint") for pair in joints]
        for group in (seed_single, seed_joint):
            identities = {(row["state_sha256"], row["python_state_sha256"], row["decision_prefix_sha256"]) for row in group}
            if len(identities) != 1:
                raise ValueError("Continuation branches differ in initial state")
        single.extend(seed_single)
        joint.extend(seed_joint)
        states.append(dict(demand_seed=seed, target=plan["target"], n_continuations=len(seed_single),
                           paired_effect=paired_summary(row["difference"] for row in seed_single),
                           delta_Q_user_R_a1_minus_R_a2=paired_summary(row["delta_Q_user_R_a1_minus_R_a2"] for row in seed_single),
                           raw_branch_trip_mean_spread={arm: paired_summary(row[f"{arm}_travel_seconds"] for row in seed_single)
                                                       for arm in ("baseline", "treatment")},
                           direct_and_spillover={key: paired_summary(row[key] for row in seed_single if row[key] is not None)
                               for field in FIELDS for key in (f"target_vehicle_difference_{field}",
                                   f"other_vehicles_total_difference_{field}", f"target_contribution_to_global_mean_{field}",
                                   f"other_contribution_to_global_mean_{field}")},
                           components={field: paired_summary(row[f"difference_{field}"] for row in seed_single) for field in FIELDS},
                           all_population_complete=all(row["all_population_complete"] for row in seed_single)))
    if not single or not joint:
        parser.exit(1, "No fully evaluated demand seeds found; no summary written.\n")
    if omitted:
        parser.exit(1, f"Incomplete demand seeds {omitted}; no final summary written.\n")
    pilot = protocol["demand_seeds"][0]
    state_means = [state["paired_effect"]["mean"] for state in states]
    positive = sum(state["paired_effect"]["ci_low"] is not None and state["paired_effect"]["ci_low"] > 0 for state in states)
    negative = sum(state["paired_effect"]["ci_high"] is not None and state["paired_effect"]["ci_high"] < 0 for state in states)
    rows = single + joint
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(), pilot_seed=pilot,
                   included_demand_seeds=sorted(plans), omitted_incomplete_demand_seeds=omitted,
                   sign_convention="positive baseline-minus-treatment means lower treatment trip time",
                   inference="descriptive paired percentile bootstrap; no multiplicity-adjusted significance claims",
                   single=summarize_rows(single, pilot), per_state=states,
                   single_state_effect_distribution=dict(n_states=len(states), minimum=float(min(state_means)),
                       median=float(np.median(state_means)), maximum=float(max(state_means)),
                       effect_mean_distribution=paired_summary(state_means), positive_interval_count=positive,
                       negative_interval_count=negative, positive_interval_fraction=positive / len(states),
                       negative_interval_fraction=negative / len(states)),
                   joint={str(dose): summarize_rows([row for row in joint if row["fraction"] == dose], pilot)
                          for dose in sorted({row["fraction"] for row in joint})},
                   completion=dict(all_branches_complete=all(row["all_population_complete"] for row in rows),
                       max_unfinished=max(max(row["baseline_unfinished"], row["treatment_unfinished"]) for row in rows),
                       denominator_range=[min(row["denominator"] for row in rows), max(row["denominator"] for row in rows)],
                       minimum_completion_rate=min(min(row["baseline_completion_rate"], row["treatment_completion_rate"]) for row in rows)),
                   departure_time_audit=dict(changed_pair_count=sum(row["actual_departure_changed_vehicle_count"] > 0 for row in rows),
                       pair_count=len(rows), maximum_changed_vehicle_count=max(row["actual_departure_changed_vehicle_count"] for row in rows),
                       mean_departure_difference_range_seconds=[min(row["mean_actual_departure_time_difference_seconds"] for row in rows),
                           max(row["mean_actual_departure_time_difference_seconds"] for row in rows)],
                       maximum_absolute_mean_departure_difference_seconds=max(abs(row["mean_actual_departure_time_difference_seconds"]) for row in rows),
                       sign_convention="actual_departure_baseline_minus_treatment; add to trip improvement for identical schedule-to-arrival improvement"),
                   provenance=dict(protocol=protocol, completed=completed, seed_discovery_plans={str(seed): plan for seed, plan in plans.items()},
                       summary_source=str(Path(__file__).relative_to(ROOT))))
    write_csv(output / "single_pairs.csv", single)
    write_csv(output / "joint_pairs.csv", joint)
    write_json(output / "summary.json", summary)
    make_report(output, protocol, completed, plans, single, joint, states, summary)
    make_plots(output, single, joint, states, summary)
    print(json.dumps(dict(output=str(output), demand_count=len(plans), single_pairs=len(single), joint_pairs=len(joint),
                         all_branches_complete=summary["completion"]["all_branches_complete"])))


if __name__ == "__main__":
    main()
