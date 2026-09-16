"""Write the Korean revision record directly from measured result manifests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "audit/lstm_comparison"
RUN = ROOT.parent / "serverless-fewshot/results/lstm_comparison_v1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(DATA/"manifest.json")
    verified = read(DATA/"verification.json")
    complete = verified["status"] == "passed" and not manifest["missing_cases"]
    build_path = ROOT/"audit/build_manifest.json"
    build = read(build_path) if build_path.exists() else {}
    final_pdf = complete and build.get("measurement_verification_sha256") == sha(DATA/"verification.json")
    rows = ["# LSTM 비교 추가 및 투고 원고 개정 보고서", "",
            "작성 기준: "+datetime.now(timezone.utc).isoformat(), "",
            "## 1. 현재 상태", "",
            f"- DES 실험: **{len(manifest['completed_cases'])}/19개 실험군 완료**.",
            f"- 원자료 검증: **{verified['status']}**.",
            f"- 과학적·문서 테스트: **{verified['tests']['total_tests']}개 통과**.",
            f"- HTTP 상태를 보존한 5개 비교군 실서비스 실험: **{verified['live']['status']}**."]
    if not complete:
        rows += ["- 전체 완료 전 기록이다. 남은 실험: "+", ".join(manifest["missing_cases"])+".",
                 "- 현재 루트 PDF는 최종 결과물로 확정되지 않았다. 조판 미리보기는 별도 디렉터리에 있다."]
    if final_pdf:
        rows += [f"- 최종 PDF: 본문 **{build['documents']['main']['pages']}쪽**, 보충자료 **{build['documents']['supplementary']['pages']}쪽**, highlights **{build['documents']['highlights']['pages']}쪽**.",
                 f"- 빌드 ID: `{build['build_id']}`. 표·그림·PDF 해시는 `build_manifest.json`에 기록했다."]
    rows += ["", "## 2. 유지한 논문의 중심", "",
             "제목과 WINTER의 TCN 표현·prototype 초기값·ridge 적응·WINTER-G 전환 규칙을 유지했다. "
             "LSTM을 추가하면서 초기 관측 구간의 CSR 및 적응 지연, 이후 장기 실행에서의 메모리 교환관계를 검증하도록 결과를 갱신했다. "
             "전체 제공자에서 최고의 예측기라는 주장이나 ANIL 자체의 필수성을 입증했다는 주장은 추가하지 않았다.", "",
             "## 3. 구현한 비교 연구", "",
             "**Gunasekaran et al., Fifer: Tackling Resource Underutilization in the Serverless Era (Middleware 2020)**. "
             "[원 논문](https://arxiv.org/abs/2008.12819), [DOI](https://doi.org/10.1145/3423211.3425683).", "",
             "| 구분 | 구현 내용 |", "|---|---|",
             "| 보존한 원리 | 과거 부하에서 미래 최대 수요 예측, 부족 용량 선행 생성, 예측 실패 시 반응형 생성, 10분 idle timeout |",
             "| 시간 해상도 | 원문의 20개 5초 표본·10초 갱신을 공개 트레이스의 20개 1분 표본·1분 갱신으로 변경; 10분 예측 구간 유지 |",
             "| 네트워크 | 64 hidden unit의 단일 LSTM + scalar head, 17,217 parameters |",
             "| 학습 | log1p 입력·peak target, Adam, source-only 학습 및 검증, target 가중치 갱신 없음 |",
             "| LSTM–Fifer | ceil(예측 peak), batch size 1, 용량 상한 200, 10분 timeout |",
             "| LSTM–shared | 같은 예측을 논문의 공통 quantile/lifetime controller에 입력 |",
             "| 구현 범위 | 독립 함수의 예측·prewarming. 원 시스템의 DAG batching, node bin packing, Brigade 배포 전체 재현은 아님 |", "",
             "아키텍처·optimizer 등 원문이 명시하지 않은 선택은 본 연구의 구현 설정으로 구분했다. "
             "Hu et al. (2025)의 RNN prewarming 연구는 관련 연구로 유지하며, 해당 시스템을 재현했다고 표기하지 않는다.", "",
             "## 4. 실험 범위", "",
             "| 실험 | 대상 | 반복 및 비용 비율 |", "|---|---|---|",
             "| 초기 4시간 | Azure 2019 100개, Huawei 76개, 추가 Azure 970개 함수 | 20 DES seeds, ρ=1/10/100 |",
             "| 연속 48시간 | Azure 2019 1,755개 함수 | 3 seeds, ρ=1/10/100 |",
             "| 활성 함수군 | Azure 2021·2019 각 S1/S2/S3, Huawei mixed/sparse/saturated | 3 seeds, ρ=0.1/1/10/100 |",
             "| 드리프트 | 제공자별 자연 180개 + 합성 960개, 총 3,420개 구간 | 24시간 burn-in + 4시간, 3 seeds, ρ=1/10 |",
             "| 실서비스 | 10개 함수 × 5개 정책, 정책당 686개 요청 | 1시간 실시간 재생, 개별 HTTP 상태 기록 |",
             "| 예측기 비용 | WINTER body/read와 LSTM | 동일 batch 1/100/1,000, CPU·GPU, 30회 측정 |", "",
             "고정 정책별 전체 계획은 1,560,042개 함수×정책×seed 조건이다. "
             "동일한 native 정책 스케줄은 비용 비율별로 중복 실행하지 않는다. 실제 실행 수는 검증 JSON에 별도로 기록된다.", "",
             "## 5. 핵심 결과 — ρ=10", "",
             "### 초기 관측 구간", "",
             "| 대상 | WINTER CSR (4h, %) | LSTM–Fifer CSR (%) | AL: WINTER / LSTM (분) | idle memory W/L | W−L CSR 95% CI (pp) |",
             "|---|---:|---:|---:|---:|---|" ]
    for case,label in [("initial_azure_primary","Azure 100"),("initial_huawei","Huawei 76"),("initial_azure_evaluation","Azure 970")]:
        d=read(DATA/(case+".json"));w,l=[d["by_action"][m+"__rho10"] for m in ("WINTER","LSTM_Fifer")]
        a,b=w["windows"]["full"],l["windows"]["full"]
        ci=d["paired_bootstrap"]["contrasts"]["WINTER_vs_LSTM_Fifer__rho10"]["full"]["winter_minus_comparator_csr_pp_ci95"]
        rows.append(f"| {label} | {a['csr_pct']:.3f} | {b['csr_pct']:.3f} | {w['adaptation_lag']} / {l['adaptation_lag']} | {a['idle_per_1k']/b['idle_per_1k']:.2f}× | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    rows += ["", "초기 4시간의 평균 CSR은 WINTER가 낮지만 추가 idle memory가 필요하다. "
             "작은 두 코호트의 cluster 신뢰구간은 0을 포함하며, 큰 Azure 코호트에서는 차이가 작아진다. "
             "AL은 정책별 마지막 1/4 구간의 CSR을 기준으로 계산하므로 CSR 및 메모리와 함께 읽는다.", "",
             "### 48시간 및 드리프트", "",
             "48시간 CSR은 WINTER-G 1.2995%, native LSTM 1.3037%, shared-controller LSTM 1.2975%이다. "
             "Gate의 native LSTM 대비 CSR 차이는 0.0042 pp이고 idle memory는 5.7% 많다. "
             "원래의 720분 전환은 계속 적응 출력을 선택하는 경우보다 메모리를 11.1% 줄인다.", "",
             "드리프트 후 4시간에는 두 Azure 데이터에서 scheduled WINTER의 CSR이 native LSTM보다 낮고 메모리는 많다. "
             "Huawei에서는 자연·합성 드리프트 모두 native LSTM이 CSR과 메모리에서 유리하다. "
             "같은 AL이 같은 CSR을 뜻하지 않는 사례도 본문에 반영했다.", "",
             "### 활성 함수군", "",
             "| 대상 | WINTER CSR (%) | LSTM CSR (%) | WINTER-G CSR (%) | idle memory LSTM / EWMA |",
             "|---|---:|---:|---:|---:|"]
    for name in manifest["expected_cases"]:
        if not name.startswith("steady_") or name not in manifest["completed_cases"]:
            continue
        d=read(DATA/(name+".json"));a={m:d["by_action"][m+"__rho10"]["windows"]["full"] for m in ("WINTER","LSTM_Fifer","WINTER_G","EWMA_0.1")}
        rows.append(f"| {name.replace('steady_','')} | {a['WINTER']['csr_pct']:.3f} | {a['LSTM_Fifer']['csr_pct']:.3f} | {a['WINTER_G']['csr_pct']:.3f} | {a['LSTM_Fifer']['idle_per_1k']/a['EWMA_0.1']['idle_per_1k']:.2f}× |")
    rows += ["", "Sparse workload에서 작은 양의 peak 예측도 ceil 연산 후 최소 1개 환경을 유지하게 한다. "
             "Huawei sparse의 낮은 native LSTM CSR과 큰 메모리는 이 정책의 실제 결과이며, target별 threshold로 보정하지 않았다."]
    live_path = DATA/"live_results.json"
    if verified["live"]["status"] == "passed" and live_path.exists():
        live = read(live_path)
        assert sha(live_path) == verified["live"]["source_sha256"]
        rows += ["", "### 실서비스 동작 확인", "",
                 "| 스케줄 | Event CSR (%) | Latency CSR (%) | Ready pod-seconds | 실패 요청 |",
                 "|---|---:|---:|---:|---:|"]
        for key,label in [("reactive","Reactive"),("keepalive10","Keep-alive"),
                          ("ewma","EWMA"),("protowarm","Learned schedule"),
                          ("lstm_fifer","LSTM–Fifer schedule")]:
            arm = live["arms"][key]
            rows.append(f"| {label} | {100*arm['csr_event']:.3f} | {100*arm['csr_latency']:.3f} | {arm['pod_seconds']:,.0f} | {arm['failed_requests']} |")
        rows += ["", "정책당 686회 요청의 개별 HTTP 상태를 보존하고 2xx만 성공으로 집계했다. "
                 "동일 노드의 50개 서비스에서 한 시간 동안 사전 계산한 스케줄을 실행했으며, DES 작업도 같은 호스트에서 진행됐다. "
                 "함수당 pod floor 상한이 1인 제한된 동작 확인이므로, DES의 작은 CSR 차이나 독립된 서비스 지연 성능을 입증하는 결과로 해석하지 않는다."]
    rows += ["", "## 6. 검증 및 데이터 구분", "",
             "- 정책 간 동일한 arrival·duration·reactive-init 표본을 사용하고, proactive-init은 별도 stream으로 분리했다.",
             "- TTL 변경은 과거 idle 시간을 다시 가격 매기지 않는다. init / execution / ready-idle과 terminal drain을 분리했다.",
             "- 별도 event queue·interval ledger와 비교하고, 대용량 실행 엔진·분할·checkpoint 재개가 같은 결과를 내는지 검증했다.",
             "- 함수별 결과에서 전체 합계·timeline·CSR·비용·AL 및 주요 paired bootstrap을 다시 계산했다.",
             "- Azure의 application cluster, Huawei의 function cluster를 사용하고 Azure 2019 활성 함수군의 표본 가중치를 유지했다.",
             "- 초기 결과의 20 seeds와 그 밖의 3 seeds는 요청 난수 반복이다. 모델 학습 seed에 대한 전체 불확실성은 아니다.",
             "- 이전 실험 원자료와 시행 전 수정·실서비스 capacity preflight 기록은 보존하고 최종 실험군에서 제외했다.", "",
             "## 7. 원고에 반영한 위치", "",
             "| 위치 | 변경 |", "|---|---|",
             "| Abstract / Introduction / Related Work | LSTM 비교 결과와 초기 관측 구간 중심 포지셔닝 |",
             "| Design / Evaluation Protocol | LSTM의 입력·학습·controller 인터페이스 및 원 논문과의 차이 |",
             "| Initial-Window | CSR 15/60/240분·AL·메모리·비용 표, operating point·누적 cold starts 그림 |",
             "| Continuous / Active Pools | 48시간 비교, 9개 활성 함수군 표·그림 |",
             "| Workload Shifts | 자연·합성 드리프트 표와 6개 패널 그림 |",
             "| Live / Overhead | 5개 arm 표, 같은 batch의 LSTM 비용 측정 |",
             "| Supplement | LSTM 상세 명세, 전체 비용 비율 표, AL reference, 기존 결과와의 구분 |",
             "| Highlights | TeX와 Word 문구 동기화 |", "",
             "기존 fig10 드리프트 PDF는 변경하지 않았으며, 새 측정은 fig19에 작성했다. "
             "본문 fig15–19와 보충 fig20의 수치는 새 replay에서 직접 생성한다.", "",
             "## 8. 결과 파일", "",
             "- [전체 수치 CSV](lstm_comparison/all_results.csv)",
             "- [입력·실행·그림 manifest](lstm_comparison/manifest.json)",
             "- [과학적 검증 결과](lstm_comparison/verification.json)",
             "- [실험 재현 안내](../../serverless-fewshot/experiments/lstm_comparison/README.md)",
             "- 원자료: `../serverless-fewshot/results/lstm_comparison_v1/` (논문 루트 기준)",
             "- 원 실험 데이터는 위 로컬 경로에 보존한다. 원고 Git 저장소의 동기화는 실험 원자료의 공개 배포나 Overleaf 업로드를 의미하지 않는다."]
    if final_pdf:
        rows += ["- [최종 본문 PDF](../main.pdf)", "- [최종 보충자료 PDF](../supplementary.pdf)",
                 "- [Highlights Word](../highlights.docx)"]
    (ROOT/"audit/LSTM_COMPARISON_REPORT_2026-09-15_KO.md").write_text("\n".join(rows)+"\n")
    print("report written; full experiment validation:",complete,"final PDF:",final_pdf)


if __name__ == "__main__":
    main()
