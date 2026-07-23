"""순수 파이프라인 단계로서의 인물 클러스터링 (전체 재군집 + cluster_id 재조정).

군집의 진실은 event 전체 임베딩(기존+신규)에 대한 HDBSCAN 재군집이다 (ADR-003, 재군집 단위=event는 ADR-007).
이 모듈은 저장소(S3 event .npz)·SQS를 모르는 순수 로직으로, 임베딩 행렬과 직전 배정을 받아

  ⓪ 근중복 행 붕괴 — 재업로드·유령 행이 만든 유사도 ≈1.0 행 그룹을 대표 1행으로 접고 결과에서
     다시 펼친다 (HDBSCAN이 복제 쌍을 인물 덩어리보다 우선해 앨범을 쌍 단위로 쪼개는 오염 방어, ADR-029)
  ① HDBSCAN 전체 재군집 (PoC 검증 이식본, cosine) — 클러스터 0개 퇴화는 연결 성분 부분 승격으로 교정 (ADR-008)
  ② 제약 강제 — 사용자 보정(must/cannot-link)이 사람 결정을 뒤집지 않게 + 같은 사진 자동
     cannot-link(같은 사진의 두 얼굴 = 타인, ADR-011)로 물리적으로 불가능한 동거를 차단
  ③ 파편 병합 — centroid 유사도가 동일 인물 수준인 클러스터 병합 (완전 연결, 단일 인물 파편화 교정, ADR 005)
  ④ 노이즈 구제 — 최근접 centroid 유사도가 충분한 노이즈 얼굴을 클러스터에 편입
  ⑤ 저신뢰 분리 — 절대 유사도·2위 마진 임계 미달 멤버를 ambiguous로 분리 (TBD #3 기본 정책)
     + 회색지대(centroid 바닥은 넘었지만 낮음) 멤버는 face-pair 증거로 재확인해 증거 없으면 축출 (ADR 020)
     + margin 구제 — 절대 유사도는 ④ 임계에 못 미쳐도 2위 군집 대비 여유가 큰 잔여 노이즈(옆얼굴·역광·
       모션)를 top1 군집에 편입 (기본 비활성 — margin_rescue_floor 주석·2026-07-23 실측 참조)
  ⑥ 2차 파편 병합 — 구제·분리로 바뀐 최종 멤버십에 ③과 같은 판정을 재적용 (ADR-010)
  ⑦ 기존 클러스터와의 overlap(Jaccard) 매칭으로 cluster_id 승계 / 신규 발급 / 은퇴
  ⑧ 클러스터별 대표벡터(L2 정규화 평균, 파생 캐시) 계산

을 수행한다 (feature-spec §4). 임베딩 로드/저장과 보정 메시지(merge/split/reassign)의
제약 변환은 호출자(워커)의 책임이다.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.hdbscan_standalone import HDBSCAN

# embed.EMBED_DIM과 같은 값 — embed를 import하면 onnxruntime·model_source(huggingface_hub)
# 임포트 체인이 순수 수학 모듈에 유입되므로 로컬 상수로 중복 선언한다 (align._ensure_bgr와 같은 이유).
EMBED_DIM = 512
_NOISE = -1


@dataclass(frozen=True)
class ClusterConfig:
  """`recluster`의 튜닝 파라미터. 기본값은 PoC(face-detection-PoC)가 검증한 레시피다.

  거리 metric은 cosine 고정으로 노출하지 않는다 — 임베딩이 L2 정규화 단위벡터라는
  파이프라인 전제(embed 후처리·대표벡터 정의)가 cosine에 결합되어 있다.
  """

  min_cluster_size: int = 2
  # PoC 검증값 2 유지. ARI 스윕(ADR-009)에서 3이 교차연령(child) ARI를 올렸으나, 자가검증 (e)가 잡아냈듯
  # n=2 소규모 이벤트에서 blob 승격이 `n >= max(mcs, min_samples)` 게이트 뒤라 실행되지 않아 2장 인물
  # 앨범이 미형성되는 회귀가 있어 기각했다 (안전한 채택엔 승격을 min_samples 게이트에서 분리하는 코드 수정 필요).
  min_samples: int = 2
  cluster_selection_epsilon: float = 0.15
  # 기존 cluster_id 승계에 필요한 최소 Jaccard (TBD feature-spec §10 #4). 대량 업로드 시
  # 신규 멤버가 많을수록 Jaccard가 자연히 낮아지므로(기존 10 + 신규 100이면 최대 0.09)
  # 기본값은 0.0 — 겹침이 하나라도 있으면 승계 후보가 되고, 최강 겹침부터 배정된다.
  min_match_jaccard: float = 0.0
  # 파편 병합 임계 — centroid 코사인 유사도가 이 이상이면 같은 인물의 파편으로 보고 병합한다.
  # 0.55 (ADR-012): 입력 품질 교정(정렬 AA + 랜드마크 정제) 후 라벨 코퍼스 분포 측정에서 타인 쌍은
  # 최고 0.4584인 반면 동일 인물 세션 간 쌍의 77%가 구 임계 0.68에 미달 — ARI 스윕 0.45~0.60 전
  # 구간에서 5인 완벽 분리(오병합 0), 0.40에서 첫 오병합. 0.55는 그 고원의 안전 중앙이다.
  # 주의: ADR-008의 "타인 centroid 최대 0.635"(face-test 아동 교차연령, 교정 전 임베딩)는 교정 후
  # 재검증되지 않았다(데이터 부재) — 아동 다수 이벤트에서 오병합이 관찰되면 .env로 즉시 복귀하고
  # face-test child 셋을 교정 후 파이프라인으로 재측정할 것 (ADR-012 §리스크).
  merge_centroid_similarity: float = 0.55
  # 파편병합 face-level 응집 바닥 — 두 파편의 모든 얼굴 쌍(파편 i × 파편 j) 코사인의 평균이 이 값
  # 이상이어야 병합한다. centroid 임계와 AND로 결합된다 (ADR-016). centroid는 평균이라 어린아이
  # 얼굴을 뭉뚱그려 서로 다른 아이도 0.55~0.63으로 붙이는데(나이대 효과, event 35 실측), 판별 신호는
  # 개별 얼굴 쌍에 남아 있다(같은 인물 파편쌍 face평균 0.65 vs 다른 아이 ≤0.50). 이 바닥이 그 갭을
  # 가른다. 실 라벨 아동 8인 셋에서 현행(비활성) ARI 0.245 → 이 게이트로 0.788, 성인·단일인물 무회귀.
  # min이 아닌 평균(mean linkage)을 쓰는 이유: 단일 하드 포즈 쌍에 강건하고 마진이 넓다. 0 = 비활성
  # (centroid만으로 판정, 기존 동작과 동일).
  # 0.475 (ADR-016 재보정 2026-07-16): 최초값 0.45는 코퍼스 meanARI 최대였으나 실 아동 이벤트
  # (43·38, 35 잔여)에서 오병합이 남았다(event 43 최대 앨범 16얼굴, 내부 타인쌍 37%). 재스윕에서
  # 0.475는 코퍼스 meanARI 0.892(0.45 대비 -0.002, child 0.784)로 사실상 무회귀이면서 실 이벤트를
  # 0.50과 동일하게 완전 분해(43 타인쌍 37%→6%, 35는 0%). 0.45→0.475에서 갈라진 조각은 전 쌍
  # cross face평균 0.40~0.47(타인 분포)로 분리 정당 확인. 0.50은 child 과분할(ARI 0.637)로 기각.
  # 초기 근거(CHMO-269, 0.45 채택 시): 비활성 대비 child ARI 0.245→0.788, 성인·단일인물 무회귀,
  # 실 이벤트 분해(27·29·33·35) 적대적 검증 전부 GOOD(cross face평균 0.31~0.44).
  merge_facepair_floor: float = 0.475
  # 파편병합 승인 검사 방식 (ADR-024) — True(기본)면 greedy 병합에서 두 컴포넌트를 '현재 전체
  # 멤버'로 재평가한다(재계산 centroid ≥ merge_centroid_similarity AND 전체 얼굴 교차 face-pair
  # 평균 ≥ merge_facepair_floor). False는 구 동작 — 병합 전 파편 스냅샷 쌍 전부의 완전 연결 요구.
  # 구 방식은 2얼굴 파편의 centroid 노이즈에 취약하다: 같은 인물 파편 둘이 주 앨범과는 각각 게이트를
  # 통과하는데 서로만 살짝 미달하면(event 90 실측 스냅샷 쌍 0.508/0.458) 나중 파편의 합류가 막혀
  # 같은 인물 앨범이 쪼개진다. 컴포넌트 전체 재평가는 이를 붙이면서도(0.641/0.476 통과) 남남이
  # 섞인 다리(bridge) 융합은 남남 얼굴 쌍이 전체 face-pair 평균을 끌어내려 여전히 차단한다(자가검증
  # (m)). 실측: 이벤트 52개 중 event 90만 치유([15,9,8,2,2]→[17,9,8,2]), 라벨 코퍼스·나머지 51개
  # ·합성 자가검증 전부 무변화. 회귀 시 .env(CLUSTER_MERGE_COMPONENT_LINKAGE=false)로 즉시 복귀.
  merge_component_linkage: bool = True
  # 노이즈 구제 임계 — 최근접 centroid 유사도가 이 이상인 노이즈 얼굴을 그 클러스터에 편입한다.
  # 동일 인물 하한(≈0.6) 수준. 1.0에 가깝게 올리면 사실상 비활성.
  rescue_similarity: float = 0.6
  # margin 구제 (2026-07-23 커스텀 15인 세트 실측) — 절대 유사도가 rescue_similarity에 못 미치는 잔여
  # 노이즈(옆얼굴·역광·모션)를 "top1 군집 유사도 ≥ floor AND top1 ≥ ratio×top2"일 때만 top1에 편입한다.
  # 절대 임계 인하로는 불가한 대역이다 — 동일인 하드 포즈↔정면(0.34~0.49)이 타인·형제 구간(0.42~0.46,
  # ADR-012 타인 최고 0.4584)과 겹친다. 판별 신호는 상대 여유에 남는다: 동일인 하드 포즈는 top1
  # 0.44~0.49 / top2 0.14~0.28(배율 1.7~3.3)인 반면 형제 모호·타인 top1은 배율 ≈1.0. floor도 하중을
  # 받는다 — 배율만 통과하는 타인 top1이 0.22~0.26 대역에 실존(배율 1.7~1.9 관측). ⑤ 축출 뒤에
  # 최종 노이즈(축출 강등분 포함, ambiguous 제외)를 재심한다 — 근거는 _recluster_core 주석.
  # 코퍼스 실측(scripts/eval_accuracy.py, floor 0.40·ratio 1.7): 성인 9인 50장 쌍 F1 0.792→0.872
  # (recall +0.118)·오답쌍 0, 라벨 아동 8인 52장 무변화(구제 0·오배정 0), 닮은꼴(카리나·고윤정)
  # 무변화 — 전 데이터셋 정밀도 1.0 유지. 알려진 한계: 닮은꼴 형제가 자기 군집 없이 본인 부재
  # 사진에 등장하면 오배정 가능(자매 top1 0.561 실측 — 같은 사진 cannot-link만 차단).
  # floor 0 = 비활성(기본 — 실 이벤트 적대 검증 전까지 실험 전용).
  margin_rescue_floor: float = 0.0
  margin_rescue_ratio: float = 1.7
  # 저신뢰 분리 임계 (TBD feature-spec §10 #3의 초기값) — 아래 둘 중 하나라도 걸리면 ambiguous로 뺀다:
  # 자기 centroid 절대 유사도 바닥, 그리고 2위 클러스터와의 유사도 마진.
  min_membership_similarity: float = 0.4
  min_membership_margin: float = 0.05
  # 저신뢰 분리의 회색지대 face-pair 재확인 게이트 (ADR 020) — LOO centroid가 바닥(0.4)은 넘었지만
  # 이 값 미만인 "회색지대" 멤버는, 클러스터 내 최강 face-pair가 아래 floor 미만이면(동일인 증거
  # 부재) 노이즈로 축출한다. 전역 바닥 상향은 불가 — 동일인 LOO와 남남 LOO가 [0.40, 0.46)에서
  # 섞인다(라벨 코퍼스 진짜 멤버 최저 0.502 vs 실 이벤트 남남 부착 0.402~0.456, event 61 실측
  # 0.425). 판별 신호는 ADR-016과 동일하게 개별 쌍에 남는다: 남남 부착의 top쌍 ≤0.44 vs 진짜
  # 멤버 top쌍 ≥0.469. ceiling 0.46 = 남남 관측 최고 0.456 직상 + 코퍼스 진짜 멤버 LOO 최저
  # 0.502 아래. 둘 중 하나라도 0이면 비활성(기존 동작).
  evict_gray_ceiling: float = 0.46
  # 회색지대 멤버의 잔류 자격 — 클러스터 내 최강 face-pair가 이 이상이면 동일인 증거로 보고 보호.
  # 0.45 = blob 승격 간선 임계와 동일값: 실측 갭 [0.44, 0.469] 안이면서, 승격 성분(전 간선 ≥0.45)의
  # 멤버가 승격 직후 이 게이트로 재강등되는 churn을 구조적으로 차단한다 (blob_promote_floor
  # 불변식과 같은 계열 — __post_init__에서 강제).
  evict_facepair_floor: float = 0.45
  # 균질 blob 부분 승격 간선 임계 — HDBSCAN이 클러스터 0개를 낸 전원 노이즈에서, 쌍 유사도가 이 이상인
  # 얼굴끼리 간선으로 이어 연결 성분을 만든다 (ADR-008). 실측 동일 인물 쌍 하한(0.46, 포즈 변화 실사진)
  # 직하이면서 타인 상한(≲0.3) 대비 넉넉한 마진 (TBD #4에서 실데이터 재조정).
  blob_promote_similarity: float = 0.45
  # 승격 성분의 완전 연결 바닥 — 성분 내 모든 쌍이 이 이상이어야 승격한다. 닮은 중간자가 두 인물을
  # 한 성분으로 잇는 체이닝 오병합 차단 (타인 상한 대비 +0.1 마진). 간선 임계 이하·저신뢰 바닥 이상이어야
  # 한다 (__post_init__ 불변식).
  blob_promote_floor: float = 0.4
  # 미매칭이 아니라 라우팅 정책 토글이다 (군집 판단엔 영향 없음, 핸들러가 결과 조립 때 읽는다):
  # 주 인물 얼굴이 2명 이상인 사진은 매칭 여부와 무관하게 공용 앨범에도 노출한다 (인물 앨범과 중복 노출,
  # feature-spec §6.2). False면 구 정책 — 전원 미매칭인 2+ 사진만 공용으로 보낸다. Spring/앱이 새
  # common_album 의미(단체 사진 전부 포함)를 감당할 준비가 될 때까지 끄고 배포하는 롤아웃 스위치.
  group_photo_to_common: bool = True
  # 위와 같은 라우팅 정책 토글: 인물 앨범에 배정된 사진이라도 주 인물 미매칭 얼굴이 남아 있으면
  # uncertain("분류가 어려워요")에도 노출한다 — 미등록 인물을 수동 편입(__uncertain__ reassign)할 진입점을
  # 연다 (인물·공용과 중복 노출, feature-spec §6.2 결정 2026-07-21). 행인·오검출·파편(주 인물 자격 미달)은
  # 종전대로 싣지 않는다. False면 구 정책 — 매칭 얼굴이 하나라도 있는 사진은 uncertain에서 제외.
  unmatched_main_to_uncertain: bool = True
  # 위 얼굴 수 카운트의 주 인물 자격 — 그 사진 최대 얼굴 폭 대비 이 비율 미만이면 지나가는 행인으로 보고
  # 세지 않는다 (quality의 blur/eye_main_face_ratio와 같은 논리·같은 값, ADR 022). 0이면 전체 얼굴을 센다.
  common_main_face_ratio: float = 0.5
  # 위 카운트의 실인물 자격 — 미배정(노이즈·저신뢰) 얼굴이 event 내 어떤 얼굴과도 유사도가 이 값 미만이면
  # 오검출(털·사물)의 쓰레기 임베딩으로 보고 세지 않는다. 주 인물 크기의 오검출이 머릿수에 들어가면
  # 1인 사진이 "주 인물 2명 단체"로 오판돼 공용 앨범에 노출된다 (event 93 퍼 후드, ADR 025).
  # 실측 빈 구간 (0.183, 0.191)에서 실인물 보호 쪽(하단)으로 채택 — 오검출 최고 0.183(퍼 후드) vs
  # 주 인물 크기 실인물 미배정 최저 0.191(고개 숙인 옆얼굴), 클러스터 배정 얼굴 최저 0.407. 0이면 비활성.
  common_face_min_similarity: float = 0.185
  # 위 카운트의 이중 검출 붕괴 (ADR-027) — 같은 사진의 두 얼굴 행이 이 값 이상 닮으면 타인 두 명이
  # 아니라 YuNet이 한 얼굴을 두 박스로 그린 것으로 보고 한 명으로 센다(폭 최대 행만 자격 유지).
  # 같은사진 자동 cannot-link의 이중 검출 안전판(handlers._SAME_FACE_SIMILARITY)과 같은 원리·같은
  # 기본값 — 그 안전판 덕에 파편 행 2개가 같은 인물 앨범에 얌전히 들어가고도, 머릿수에는 이 붕괴가
  # 없어 1인 셀피가 "주 인물 2명 단체"로 공용 앨범에 노출됐다(event 105). 실측(24개 event 같은사진
  # 쌍 758개): 이중 검출 전부 0.978~0.979 vs 같은사진 타인 쌍 최고 0.756 — 0.95가 빈 구간 안.
  # 0이면 비활성(기존 동작 — 행 수 그대로 센다).
  common_duplicate_face_similarity: float = 0.95
  # 재군집 입력의 근중복 행 붕괴 (ADR-029) — 두 행이 이 값 이상 닮으면 같은 원본의 복제(같은 사진
  # 재업로드가 새 image_id로 재임베딩, 또는 delete_request 미도달 유령 행)로 보고 재군집 전에 대표
  # 1행으로 접는다(결과 조립 시 전 행으로 펼침). HDBSCAN은 밀도 기반이라 거리 0의 복제 쌍을 느슨한
  # 실제 인물 덩어리(쌍 0.39~0.67)보다 안정적인 클러스터로 선택해 앨범이 쌍 단위로 와해된다
  # (실 event 8, 2026-07-11 리뷰). 0.985 = 재업로드 완전 복제(1.0)와 크로스이벤트 재업로드 링크
  # 관례(≥0.985, 2026-07-22 측정)를 포괄하면서, 같은 사진 이중 검출(실측 0.978~0.979 — ADR-027
  # 붕괴의 몫)과 실서버 동일 인물 새-세션 최고(0.934)는 아래에 남는 빈 구간 값. 0이면 비활성.
  duplicate_collapse_similarity: float = 0.985
  # uncertain 사진의 품질 원인(UncertainImage.causes) 판정 임계 — 군집엔 영향 없고 핸들러가 결과 조립 때
  # 읽는다 (CHMO-404). 이 사진 주 얼굴(counted 최대 폭)이 이 px 미만이면 small_faces, 그중 원본 긴 변이
  # uncertain_low_res_long_side 미만이면 low_resolution도 함께 실린다. 실측 근거: 얼굴폭 매칭 무릎 ~100px
  # (이벤트 중앙 얼굴폭 <100px면 배정률 20~40%, ≥450px면 70~100%), 단체사진 얼굴 rel_w 중앙 ~5%라
  # 100px 얼굴엔 긴 변 ~2000px 필요(≤1080 이벤트 배정률 20%, ≥4896 이벤트 70~100%). 0이면 비활성.
  uncertain_small_face_px: float = 100.0
  uncertain_low_res_long_side: float = 2000.0

  def __post_init__(self) -> None:
    # 이식한 HDBSCAN이 min_cluster_size < 2에서 raise하므로 생성 시점에 같은 계약을 강제한다
    if self.min_cluster_size < 2:
      raise ValueError(f"min_cluster_size는 2 이상이어야 합니다. 받은 값: {self.min_cluster_size}")
    if self.min_samples < 1:
      raise ValueError(f"min_samples는 1 이상이어야 합니다. 받은 값: {self.min_samples}")
    if self.cluster_selection_epsilon < 0.0:
      raise ValueError(f"cluster_selection_epsilon은 0 이상이어야 합니다. 받은 값: {self.cluster_selection_epsilon}")
    for name in (
      "min_match_jaccard",
      "merge_centroid_similarity",
      "merge_facepair_floor",
      "rescue_similarity",
      "margin_rescue_floor",
      "min_membership_similarity",
      "min_membership_margin",
      "evict_gray_ceiling",
      "evict_facepair_floor",
      "blob_promote_similarity",
      "blob_promote_floor",
      "common_main_face_ratio",
      "common_face_min_similarity",
      "common_duplicate_face_similarity",
      "duplicate_collapse_similarity",
    ):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}은(는) [0, 1] 범위여야 합니다. 받은 값: {value}")
    for name in ("uncertain_small_face_px", "uncertain_low_res_long_side"):
      value = getattr(self, name)  # px 임계라 [0,1]이 아니다 — 0(비활성) 이상의 유한값만 강제
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}은(는) 0 이상의 유한값이어야 합니다. 받은 값: {value}")
    if self.cluster_selection_epsilon > 2.0:
      # cosine 거리 범위는 [0, 2] — 밖의 값은 기하학적으로 무의미한데도 조용히 동작해 군집 선택을 왜곡한다
      raise ValueError(
        f"cluster_selection_epsilon은 cosine 거리 범위 [0, 2] 안이어야 합니다. 받은 값: {self.cluster_selection_epsilon}"
      )
    if self.margin_rescue_floor > 0 and self.margin_rescue_ratio < 1.0:
      # 배율 1 미만이면 top2가 top1에 근접해도(형제 모호) 통과한다 — 게이트의 존재 이유가 무너지는
      # 오설정을 즉시 드러낸다
      raise ValueError(f"margin_rescue_ratio는 1 이상이어야 합니다. 받은 값: {self.margin_rescue_ratio}")
    if self.rescue_similarity < self.min_membership_similarity:
      # 이 순서가 깨지면 [rescue, floor) 대역에서 구제된 얼굴이 같은 실행의 저신뢰 축출에서 곧바로
      # 노이즈로 재강등된다 — 구제 시점과 축출 시점의 자기 제외 유사도가 정확히 같기 때문 (리뷰 재현).
      raise ValueError(
        "rescue_similarity는 min_membership_similarity 이상이어야 합니다. "
        f"받은 값: rescue={self.rescue_similarity}, floor={self.min_membership_similarity}"
      )
    if self.blob_promote_floor > self.blob_promote_similarity:
      # 간선(≥ promote)으로 이어진 쌍이 floor를 자동 충족해야 성분 완전 연결 검사의 의미가 성립한다
      raise ValueError(
        "blob_promote_floor는 blob_promote_similarity 이하여야 합니다. "
        f"받은 값: floor={self.blob_promote_floor}, promote={self.blob_promote_similarity}"
      )
    if self.blob_promote_floor < self.min_membership_similarity:
      # 승격 성분 멤버의 LOO 유사도는 성분 쌍 유사도 평균 이상(‖타멤버 합‖ ≤ n-1)이라 이 순서가 지켜지면
      # 승격된 멤버가 같은 실행의 저신뢰 축출(절대 바닥)에서 곧바로 노이즈로 재강등되지 않는다
      # (rescue_similarity 불변식과 같은 계열의 churn 차단).
      raise ValueError(
        "blob_promote_floor는 min_membership_similarity 이상이어야 합니다. "
        f"받은 값: blob_floor={self.blob_promote_floor}, membership_floor={self.min_membership_similarity}"
      )
    if self.evict_gray_ceiling > 0 and self.evict_facepair_floor > 0:
      if self.evict_gray_ceiling < self.min_membership_similarity:
        # ceiling이 바닥 아래면 회색지대가 공집합이라 게이트가 조용히 죽는다 — 오설정을 즉시 드러낸다
        raise ValueError(
          "evict_gray_ceiling은 min_membership_similarity 이상이어야 합니다. "
          f"받은 값: ceiling={self.evict_gray_ceiling}, membership_floor={self.min_membership_similarity}"
        )
      if self.evict_facepair_floor > self.blob_promote_similarity:
        # 승격 성분의 간선(≥ promote_similarity)이 facepair floor를 자동 충족해야, 방금 승격된
        # 성분이 같은 실행의 회색지대 게이트로 곧바로 해체되는 churn이 없다 (위 불변식들과 같은 계열).
        raise ValueError(
          "evict_facepair_floor는 blob_promote_similarity 이하여야 합니다. "
          f"받은 값: facepair_floor={self.evict_facepair_floor}, promote={self.blob_promote_similarity}"
        )


@dataclass(frozen=True)
class Constraints:
  """사용자 보정(병합/분리/이동)을 임베딩 행 인덱스 쌍으로 표현한 제약.

  보정 메시지(cluster-feedback의 merge/split/reassign) → 인덱스 쌍 변환은 호출자의 책임이다.
  must-link로 (전이적으로) 연결된 두 얼굴 사이의 cannot-link는 모순이라 `recluster`가
  ValueError로 거부한다 — 보정 간 충돌의 시간순 해소(나중 결정 우선)는 보정 이력을 아는
  워커 계층에서 끝내고, 이 모듈에는 일관된 제약 셋만 전달해야 한다.

  auto_cannot_link는 사람 보정이 아니라 사실에서 유도된 제약이다 (같은 사진의 두 얼굴 = 서로
  다른 사람, ADR-011). 병합 차단·클러스터 분리·구제 차단에는 cannot_link와 동일하게 참여하지만,
  "사람이 직접 지목한 얼굴" 대우는 받지 않는다 — 저신뢰 축출 보호와 축출 마진 비교 제외에 불참.
  이를 cannot_link 채널에 섞으면 전 얼굴이 축출 보호를 받아 HDBSCAN이 밀도 없는 데이터에서 낸
  무의미한 클러스터(쌍 유사도 0.2대)가 걸러지지 않는 것이 실 event 시뮬레이션에서 재현됐다.
  must-link와 모순되는 자동 쌍은 거부하지 않고 탈락시킨다 (사람 결정 우선 — 동일 사진 중복 업로드
  오염 같은 예외에서 재군집이 죽지 않아야 한다).
  """

  must_link: tuple[tuple[int, int], ...] = ()
  cannot_link: tuple[tuple[int, int], ...] = ()
  auto_cannot_link: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PersonCluster:
  """재군집 결과의 인물 클러스터 1개."""

  cluster_id: str
  is_new: bool  # 이번 재조정에서 기존 cluster_id 승계에 실패해 새로 발급된 인물인지
  member_indices: tuple[int, ...]  # 입력 embeddings의 행 인덱스 (오름차순)
  # 멤버별 자기 클러스터 신뢰도 — leave-one-out centroid 코사인 유사도, 저신뢰 축출과 동일 정의
  # (member_indices와 자리 대응, 단독 멤버는 1.0). 워커가 uncertain reason·저신뢰 표시
  # (feature-spec §6.2·§7·TBD #2)에 쓸 값을 재계산 없이 노출한다 — 외부 재계산은 LOO 보정·
  # cannot-link 마진 제외가 빠져 축출 결정과 어긋난 신뢰도를 만든다 (리뷰 지적).
  membership_similarities: tuple[float, ...]
  # 조회·표시용 파생 캐시(멤버 임베딩의 L2 정규화 평균) — 군집 판단의 원천이 아니다 (ADR-003).
  # ndarray 필드의 자동 __eq__는 진리값 모호성으로 예외를 던지므로 비교 대상에서 제외한다 (DetectedFace와 동일).
  centroid: np.ndarray = field(compare=False)  # shape (EMBED_DIM,), float32


@dataclass(frozen=True)
class ReclusterResult:
  """`recluster` 1회 실행의 결과 — 결과 메시지(classify-result)와 event .npz 갱신의 원천."""

  clusters: tuple[PersonCluster, ...]  # 최소 멤버 인덱스 오름차순
  # 어느 인물에도 배정되지 않은 얼굴 (uncertain 후보) — 밀도 노이즈(구제 실패)뿐 아니라
  # 저신뢰 절대 바닥 미달로 클러스터에서 강등된 얼굴도 포함한다
  noise_indices: tuple[int, ...]
  ambiguous_indices: tuple[
    int, ...
  ]  # 인물 배정이 저신뢰(절대 유사도·마진 미달)라 분리된 얼굴 (uncertain 'ambiguous' 후보)
  retired_cluster_ids: tuple[str, ...]  # 이번 재군집에서 승계되지 못한 기존 cluster_id


class _UnionFind:
  """must-link 컴포넌트 계산용 union-find (경로 압축 + 크기 기준 합치기)."""

  def __init__(self, n: int) -> None:
    self._parent = list(range(n))
    self._size = [1] * n

  def find(self, x: int) -> int:
    root = x
    while self._parent[root] != root:
      root = self._parent[root]
    while self._parent[x] != root:  # 경로 압축
      self._parent[x], x = root, self._parent[x]
    return root

  def union(self, a: int, b: int) -> None:
    root_a, root_b = self.find(a), self.find(b)
    if root_a == root_b:
      return
    if self._size[root_a] < self._size[root_b]:
      root_a, root_b = root_b, root_a
    self._parent[root_b] = root_a
    self._size[root_a] += self._size[root_b]


def _validate_pairs(pairs: tuple[tuple[int, int], ...], n: int, kind: str) -> None:
  for i, j in pairs:
    if not (0 <= i < n and 0 <= j < n):
      raise ValueError(f"{kind} 제약 인덱스가 범위를 벗어났습니다. 받은 쌍: ({i}, {j}), 임베딩 수: {n}")


def _must_link_components(n: int, constraints: Constraints) -> tuple[list[int], dict[int, list[int]]]:
  """must-link 폐포(전이적 연결)를 계산하고 cannot-link와의 모순을 거부한다.

  반환: (각 인덱스의 컴포넌트 루트, 루트 → 멤버 오름차순 목록). 단독 얼굴도 자기 컴포넌트를 가진다.
  """
  uf = _UnionFind(n)
  for i, j in constraints.must_link:
    uf.union(i, j)
  comp_of = [uf.find(i) for i in range(n)]
  components: dict[int, list[int]] = {}
  for idx in range(n):
    components.setdefault(comp_of[idx], []).append(idx)
  for i, j in constraints.cannot_link:
    if i == j:
      # 일반 모순 검사(comp_of[i] == comp_of[j])에 맡기면 존재하지 않는 must-link를 탓하는
      # 오도성 메시지가 나간다 (리뷰 재현) — 보정 메시지 번역 버그는 원인 그대로 알려준다
      raise ValueError(f"cannot-link 쌍은 서로 다른 얼굴이어야 합니다. 받은 쌍: ({i}, {j})")
    if comp_of[i] == comp_of[j]:
      raise ValueError(f"모순된 제약입니다: must-link로 연결된 얼굴 쌍 ({i}, {j})에 cannot-link가 지정되었습니다.")
  return comp_of, components


def _enforce_must_link(
  labels: np.ndarray,
  components: dict[int, list[int]],
  next_label: int,
  cannot_link: tuple[tuple[int, int], ...] = (),
) -> int:
  """must-link 컴포넌트 전원을 같은 라벨로 강제한다 (labels 제자리 수정).

  대상 라벨은 컴포넌트 내 비노이즈 다수결(동률 시 작은 라벨). 전원 노이즈면 새 라벨을 발급한다
  — 사용자가 같은 인물이라고 확정한 그룹은 밀도와 무관하게 클러스터로 승격한다.
  반환: 다음 합성 라벨 번호.

  후보에서 cannot-link가 금지한 라벨(컴포넌트 멤버의 cannot-link 상대가 현재 가진 라벨)은 제외한다.
  reassign의 컴포넌트는 {이동 얼굴, 목적지 대표} 2명뿐이라 다수결이 항상 1:1 동률인데, "작은 라벨"이
  HDBSCAN 내부 번호라는 우연으로 출처 라벨을 고르면 목적지 대표가 자기 클러스터에서 끌려나온다 —
  뒤이은 _enforce_cannot_link는 그 쌍만 통째로 떼어낼 수 있어 목적지 앨범이 쪼개지고 신규 id가
  발급된다 (docs/reviews/2026-07-10-reassign-mustlink-tiebreak.md 재현). 이동 방향은 쌍에 없지만
  cannot-link("출처에 있으면 안 됨")에 보존되어 있으므로, 금지 라벨 제외가 그 방향을 복원한다 —
  rescue/blob floor 불변식과 같은 계열의 "다음 단계가 곧바로 되돌릴 선택 금지"다.
  """
  partners = _cannot_link_partners(cannot_link)
  for root in sorted(components):  # 루트 순서 고정 — 새 라벨 발급 순서의 결정성
    members = components[root]
    if len(members) < 2:
      continue
    forbidden = {
      int(labels[partner]) for idx in members for partner in partners.get(idx, ()) if labels[partner] != _NOISE
    }
    member_labels = labels[members]
    non_noise = member_labels[member_labels != _NOISE]
    allowed = non_noise[~np.isin(non_noise, list(forbidden))] if forbidden else non_noise
    if allowed.size:
      values, counts = np.unique(allowed, return_counts=True)  # unique는 오름차순 → argmax 동률 시 작은 라벨
      target = int(values[np.argmax(counts)])
    else:
      target = next_label
      next_label += 1
    labels[members] = target
  return next_label


def _normalized_mean(embeddings: np.ndarray, members: Sequence[int]) -> np.ndarray:
  """멤버 임베딩의 L2 정규화 평균 — 대표벡터 정의 (feature-spec §4 ⑤).

  단위벡터들의 평균은 정확히 대척(antipodal)일 때만 0이 되어 사실상 발생하지 않지만,
  0 나눗셈 대신 첫 멤버 임베딩으로 결정적 폴백한다.
  """
  mean = embeddings[list(members)].mean(axis=0)
  norm = float(np.linalg.norm(mean))
  if norm == 0.0:
    return embeddings[members[0]].astype(np.float32)
  return (mean / norm).astype(np.float32)


def _loo_similarities(embeddings: np.ndarray, members: Sequence[int]) -> np.ndarray:
  """멤버별 자기 클러스터 leave-one-out centroid 코사인 유사도 — 신뢰도의 공통 정의.

  자기가 포함된 centroid는 유사도를 부풀려 경계 얼굴이 신뢰도 검사를 통과해 버리므로 자기를 뺀
  평균과 비교한다. 저신뢰 축출 판정과 결과(PersonCluster.membership_similarities) 노출이 같은
  함수를 쓰게 해 두 값이 어긋나지 않게 한다. 단독 멤버와 퇴화(LOO 합이 영벡터)는 판단 불능이라
  1.0(유지)으로 둔다.
  """
  block = embeddings[list(members)]
  if len(members) < 2:
    return np.ones(len(members), dtype=np.float64)
  loo = block.astype(np.float64).sum(axis=0) - block
  norms = np.linalg.norm(loo, axis=1)
  sims = np.einsum("ij,ij->i", block.astype(np.float64), loo)
  safe_norms = np.where(norms == 0.0, 1.0, norms)
  return np.where(norms == 0.0, 1.0, sims / safe_norms)


def _enforce_cannot_link(
  labels: np.ndarray,
  comp_of: list[int],
  components: dict[int, list[int]],
  cannot_link: tuple[tuple[int, int], ...],
  embeddings: np.ndarray,
  next_label: int,
) -> None:
  """같은 클러스터에 남은 cannot-link 쌍을 분리한다 (labels 제자리 수정, TBD #5의 기본 정책).

  이동 단위는 must-link 컴포넌트(또는 단독 얼굴)라 컴포넌트가 쪼개지지 않는다. 위반 쌍에 관여한
  컴포넌트(앵커)들을 greedy 그래프 컬러링으로 최소한만 갈라(제약 없는 앵커끼리는 과분리하지 않음),
  가장 큰 앵커 무리가 원 라벨을 유지하고 나머지 색은 새 라벨을 받는다. 제약에 안 걸린 나머지 멤버는
  컴포넌트 단위로 코사인 최근접 앵커 대표벡터를 따라간다. 앵커 처리 순서(크기 내림차순 → 최소 인덱스)와
  라벨 오름차순 순회로 결과는 결정적이다.

  next_label은 여기서 발급할 합성 라벨의 시작값일 뿐, 진행된 카운터를 반환하지 않는다 — 반환하면
  버리는 호출부가 생기고, 그 오래된 카운터로 라벨을 발급하는 후속 단계가 여기서 만든 라벨과
  조용히 충돌한다 (리뷰 지적). 라벨을 발급하는 후속 단계를 추가하려면 카운터가 아니라
  `labels.max() + 1`에서 다시 시작할 것.
  """
  if not cannot_link:
    return
  # 처리 중 이동은 처리 대상 라벨 안에서 새 라벨로만 일어나므로(기존 라벨로 유입 없음),
  # 위반 라벨 집합을 처음 한 번만 계산해도 안전하다. 노이즈(-1)는 클러스터가 아니라 위반이 아니다.
  violated_labels = sorted({int(labels[i]) for i, j in cannot_link if labels[i] == labels[j] and labels[i] != _NOISE})
  for current in violated_labels:
    member_idx = [int(i) for i in np.flatnonzero(labels == current)]
    pairs = [(i, j) for i, j in cannot_link if labels[i] == current and labels[j] == current]

    # 앵커 = 위반 쌍에 관여한 컴포넌트 전체 (must-link 강제 이후 컴포넌트는 라벨 균일)
    anchors: dict[int, list[int]] = {}
    adjacency: dict[int, set[int]] = {}
    for i, j in pairs:
      root_i, root_j = comp_of[i], comp_of[j]
      anchors.setdefault(root_i, components[root_i])
      anchors.setdefault(root_j, components[root_j])
      adjacency.setdefault(root_i, set()).add(root_j)
      adjacency.setdefault(root_j, set()).add(root_i)

    # greedy 컬러링: cannot-link로 인접한 앵커만 다른 색 — 색 0(가장 큰 앵커 우선)이 원 라벨 유지
    ordered_roots = sorted(anchors, key=lambda root: (-len(anchors[root]), anchors[root][0]))
    color: dict[int, int] = {}
    for root in ordered_roots:
      used = {color[neighbor] for neighbor in adjacency[root] if neighbor in color}
      chosen = 0
      while chosen in used:
        chosen += 1
      color[root] = chosen
    color_label = {0: current}
    for extra in range(1, max(color.values()) + 1):
      color_label[extra] = next_label
      next_label += 1
    for root in ordered_roots:
      labels[anchors[root]] = color_label[color[root]]

    # 제약 없는 나머지 멤버는 컴포넌트 단위로 코사인 최근접 앵커를 따라간다 — split된 인물 양쪽이
    # 이후 업로드에서도 각자 사진을 이어받을 수 있게 하기 위함이다 (원 라벨 고정 시 한쪽만 성장).
    anchor_members = {idx for members in anchors.values() for idx in members}
    units: dict[int, list[int]] = {}
    for idx in member_idx:
      if idx not in anchor_members:
        units.setdefault(comp_of[idx], []).append(idx)
    if not units:
      continue
    anchor_centroids = np.stack([_normalized_mean(embeddings, anchors[root]) for root in ordered_roots])
    anchor_labels = [color_label[color[root]] for root in ordered_roots]
    for members in units.values():
      similarities = anchor_centroids @ embeddings[members].mean(axis=0)
      labels[members] = anchor_labels[int(np.argmax(similarities))]  # argmax 동률 시 앞선(큰) 앵커


def _cannot_link_partners(cannot_link: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
  """얼굴 → cannot-link 상대 목록. 대다수인 비제약 얼굴의 차단 검사가 O(전체 쌍) 스캔 대신 O(1)이 된다."""
  partners: dict[int, list[int]] = {}
  for a, b in cannot_link:
    partners.setdefault(a, []).append(b)
    partners.setdefault(b, []).append(a)
  return partners


def _sets_blocked(set_a: set[int], set_b: set[int], cannot_link: tuple[tuple[int, int], ...]) -> bool:
  """두 멤버 집합 사이에 cannot-link 쌍이 걸쳐 있는지 (병합 차단 판정)."""
  for a, b in cannot_link:
    if (a in set_a and b in set_b) or (a in set_b and b in set_a):
      return True
  return False


def _cluster_groups(labels: np.ndarray) -> list[tuple[int, list[int]]]:
  """비노이즈 라벨별 멤버 목록을 최소 멤버 인덱스 순으로 반환한다 (합성 라벨 번호 무관 결정성)."""
  groups: dict[int, list[int]] = {}
  for idx, label in enumerate(labels):
    if label != _NOISE:
      groups.setdefault(int(label), []).append(idx)
  return sorted(groups.items(), key=lambda item: item[1][0])


def _promote_single_blob(labels: np.ndarray, embeddings: np.ndarray, config: ClusterConfig) -> None:
  """HDBSCAN이 클러스터를 하나도 못 만들었을 때, 동일 인물 수준으로 닮은 연결 성분만 골라 승격한다.

  allow_single_cluster=False에서 event 전체가 사실상 단일 군집이면 두 갈래로 깨진다: 파편화되거나
  (파편 병합이 교정), 분할 지점이 아예 없으면 클러스터 0개(전원 노이즈)가 된다 — 후자는 병합·구제가
  손댈 클러스터가 없어 인물 앨범이 아예 생기지 않는다. allow_single_cluster=True는 해법이 아니다 —
  루트 소속 판정이 epsilon(0.15, 유사도 0.85 이내)을 요구해 실사진 분산에서 오히려 전원 노이즈가
  되는 것이 실험으로 확인됐다 (ADR-005).

  이전 구현(전 쌍별 유사도 ≥ merge_centroid_similarity(0.7)일 때 전체 일괄 승격)은 근중복 버스트만
  구제했다 — 포즈 변화가 있는 실사진의 동일 인물 쌍 유사도는 0.46~0.70이라, 소규모 단일 인물
  이벤트가 전원 uncertain이 되어 인물 앨범이 생기지 않는 것이 face-test 검증에서 확인됐다 (ADR-008).

  2단 구조로 판정한다: 쌍 유사도 ≥ blob_promote_similarity 간선으로 연결 성분을 만들고, 각 성분에서
  모든 쌍 ≥ blob_promote_floor(완전 연결)인 최대 부분집합(크기 ≥ min_cluster_size)을 승격한다.
  - floor는 닮은 중간자 체이닝(A~B, B~C인데 A·C는 남남)이 두 인물을 한 성분으로 잇는 오병합을
    차단한다 (타인 쌍 ≲0.3 < floor). 간선보다 낮은 floor는 비인접 약한 쌍에만 관용을 준다.
  - 완전 연결이 깨지면 성분을 통째 버리지 않고, 연결이 가장 약한 멤버(성분 내 유사도 합 최소)를
    하나씩 떼며 재검사한다(peel). 한 장의 극단적 포즈가 성분 전체를 무너뜨리던 문제(실사진 karina
    5장에서 한 쌍이 0.394로 floor 미달 → 전원 uncertain)를 salvage한다. 승격되는 클러스터는 항상
    완전 연결 ≥ floor를 만족하므로(peel 불변식) 타인 쌍(<floor)이 같은 앨범에 남는 일은 없다.
  - 부분 승격이라 "단일 인물 + 낯선 행인" 이벤트에서 행인은 노이즈로 남고, peel로 떨어진 극단적
    포즈 얼굴도 노이즈로 남는다(전체 일괄 승격이면 오병합되거나 전체가 기각된다). 떨어진 얼굴은
    이후 rescue_similarity(노이즈 구제)가 신규 클러스터 centroid 기준으로 재편입을 시도한다.
  - 전원 노이즈 전제는 다인물 이벤트 순도의 방벽이므로 유지한다 — HDBSCAN이 클러스터를 만든
    이벤트의 잔여 노이즈는 rescue_similarity(노이즈 구제)의 몫이다.
  이후 must/cannot-link 강제는 승격된 라벨 위에서 정상 동작한다 (호출 순서: 승격 → 제약).
  """
  if labels.size == 0 or (labels != _NOISE).any():
    return
  # 전원 노이즈일 때만 계산 — 실사용에서 이 경로는 소규모 blob이고, N² 행렬은 HDBSCAN이 이미 만든 규모다
  gram = embeddings @ embeddings.T
  uf = _UnionFind(labels.size)
  for i, j in zip(*np.nonzero(np.triu(gram >= config.blob_promote_similarity, k=1))):
    uf.union(int(i), int(j))
  components: dict[int, list[int]] = {}
  for idx in range(labels.size):
    components.setdefault(uf.find(idx), []).append(idx)
  next_label = 0
  for members in sorted(components.values(), key=lambda group: group[0]):  # 최소 멤버 인덱스 순 — 라벨 결정성
    kept = members
    while len(kept) >= config.min_cluster_size:
      block = gram[np.ix_(kept, kept)]  # 대각(자기 유사도=1.0)은 floor를 항상 통과하므로 min=최소 쌍유사도
      if float(block.min()) >= config.blob_promote_floor:
        labels[kept] = next_label  # 완전 연결 부분집합 — 승격
        next_label += 1
        break
      # 완전 연결 위반 — 연결이 가장 약한 멤버(유사도 합 최소, 동률은 최소 인덱스)를 떼고 재검사
      drop = int(np.argmin(block.sum(axis=1)))
      kept = kept[:drop] + kept[drop + 1 :]


def _merge_fragments(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  threshold: float,
  facepair_floor: float = 0.0,
  component_linkage: bool = True,
) -> None:
  """centroid 유사도(threshold) AND 파편 간 face-pair 평균(facepair_floor)이 동일 인물 수준인 클러스터끼리
  병합한다 (labels 제자리 수정).

  allow_single_cluster=False 특성상 한 인물 위주의 밀집이 파편화되는 케이스(ADR 005)와 일반적인
  과분할을 함께 교정한다. cannot-link로 연결된 클러스터 쌍은 병합하지 않는다(사용자 분리 결정 보존).
  후보는 파편 쌍 게이트 통과 쌍이고, 유사도 내림차순 greedy에 병합 컴포넌트의 대표 라벨을 최소 멤버
  인덱스 클러스터로 고정해 결과가 결정적이다. 후보 유사도는 병합 전 centroid 스냅샷 기준이다.

  병합 승인 검사 — 쌍별 검사만 하면 전이 체인(A~B, B~C)이 서로 타인인 A와 C(유사도 ~0.1)를 한
  앨범으로 융합하는 것이 리뷰에서 재현됐다. 융합 차단 방식은 component_linkage로 갈린다 (ADR-024):
  - True(기본): 두 컴포넌트의 '현재 전체 멤버'로 재평가 — 재계산 centroid ≥ threshold AND 전체
    얼굴 교차 face-pair 평균 ≥ facepair_floor. 남남이 섞인 다리(bridge) 컴포넌트는 남남 얼굴 쌍이
    전체 평균을 끌어내려 바닥에 걸린다. 구 방식이 2얼굴 파편의 노이즈 centroid 쌍에 걸려 같은
    인물 파편의 합류를 막던 문제(event 90: 스냅샷 쌍 0.508/0.458로 미달, 컴포넌트 전체로는
    0.641/0.476 통과)를 해소한다.
  - False(구 동작): 완전 연결(complete linkage) — 두 컴포넌트의 모든 구성 파편 쌍이 병합 전
    스냅샷 기준으로 게이트 이상이어야 한다.

  facepair_floor > 0이면 centroid에 더해 face-level 응집을 요구한다 (ADR-016): centroid는 평균이라
  어린아이 얼굴을 뭉뚱그려 서로 다른 아이도 임계를 넘기는데, 판별 신호는 개별 얼굴 쌍에 남아 있어
  파편 i × 파편 j 전 얼굴 쌍 코사인의 평균으로 되살린다. 두 조건의 AND라 병합을 더 엄격하게만 만든다
  (새 오병합 도입 불가). facepair_floor == 0이면 centroid만으로 판정 — 기존 동작과 완전 동일.
  """
  ordered = _cluster_groups(labels)
  if len(ordered) < 2:
    return
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  similarities = centroids @ centroids.T
  if facepair_floor > 0.0:
    # 파편 쌍별 face-pair 평균 — 파편 i의 모든 얼굴 × 파편 j의 모든 얼굴 코사인의 평균 (단위벡터 = 내적)
    facepair = np.ones_like(similarities)
    for i in range(len(ordered)):
      for j in range(i + 1, len(ordered)):
        cross = embeddings[ordered[i][1]] @ embeddings[ordered[j][1]].T
        facepair[i, j] = facepair[j, i] = float(cross.mean())
  else:
    facepair = None

  def mergeable(i: int, j: int) -> bool:
    if similarities[i, j] < threshold:
      return False
    return facepair is None or facepair[i, j] >= facepair_floor

  candidates = [
    (-float(similarities[i, j]), i, j)
    for i in range(len(ordered))
    for j in range(i + 1, len(ordered))
    if mergeable(i, j)
  ]
  if not candidates:
    return

  parent = list(range(len(ordered)))

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def component_mergeable(members_i: set[int], members_j: set[int]) -> bool:
    a, b = sorted(members_i), sorted(members_j)
    if float(_normalized_mean(embeddings, a) @ _normalized_mean(embeddings, b)) < threshold:
      return False
    if facepair_floor > 0.0 and float((embeddings[a] @ embeddings[b].T).mean()) < facepair_floor:
      return False
    return True

  merged_members = {pos: set(members) for pos, (_, members) in enumerate(ordered)}
  merged_positions = {pos: {pos} for pos in range(len(ordered))}  # 완전 연결 검사용 구성 클러스터 위치 (구 동작)
  for _, i, j in sorted(candidates):
    root_i, root_j = find(i), find(j)
    if root_i == root_j:
      continue
    if _sets_blocked(merged_members[root_i], merged_members[root_j], cannot_link):
      continue
    if component_linkage:
      if not component_mergeable(merged_members[root_i], merged_members[root_j]):
        continue  # 컴포넌트 전체 재평가 미달 — 남남이 섞였으면 전체 face-pair 평균이 바닥에 걸린다 (ADR-024)
    elif not all(mergeable(p, q) for p in merged_positions[root_i] for q in merged_positions[root_j]):
      continue  # 완전 연결 위반 — 다리(bridge) 클러스터를 통한 타인/타아동 융합 차단
    if root_j < root_i:  # 작은 위치가 루트 — 컴포넌트 라벨이 최소 멤버 인덱스 클러스터로 수렴
      root_i, root_j = root_j, root_i
    parent[root_j] = root_i
    merged_members[root_i] |= merged_members.pop(root_j)
    merged_positions[root_i] |= merged_positions.pop(root_j)

  for pos, (_, members) in enumerate(ordered):
    root = find(pos)
    if root != pos:
      labels[members] = ordered[root][0]


def _rescue_noise(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  threshold: float,
) -> None:
  """최근접 centroid 유사도가 threshold 이상인 노이즈 얼굴을 그 클러스터에 편입한다 (labels 제자리 수정).

  파편 병합 뒤에 실행해 병합된 centroid를 기준으로 삼는다 (centroid는 시작 시점 스냅샷).
  (얼굴, 클러스터) 후보를 전역 유사도 내림차순으로 처리한다 — 얼굴별 인덱스 순 처리는 cannot-link
  경합(서로 배타인 두 노이즈 얼굴이 같은 클러스터를 원할 때) 시 유사도가 낮은 쪽이 자리를 선점하는
  역전이 리뷰에서 재현됐다. 전역 내림차순에서는 더 나은 매치가 항상 먼저 배정되고, 결과는 결정적이다
  (동률은 얼굴 인덱스 → 클러스터 순). cannot-link 상대가 있는 클러스터는 건너뛰고 다음 후보를 본다.
  must-link로 묶인 전원-노이즈 컴포넌트는 이미 클러스터로 승격됐으므로 여기 도달하는 노이즈는
  전부 제약상 단독 얼굴이다.
  """
  noise_idx = [int(i) for i in np.flatnonzero(labels == _NOISE)]
  ordered = _cluster_groups(labels)
  if not noise_idx or not ordered:
    return
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  similarities = embeddings[noise_idx] @ centroids.T  # (노이즈 수, 클러스터 수) — 얼굴별 GEMV 대신 1회 GEMM
  candidates = sorted(
    (-float(similarities[row, pos]), idx, pos)
    for row, idx in enumerate(noise_idx)
    for pos in range(len(ordered))
    if similarities[row, pos] >= threshold
  )
  partners = _cannot_link_partners(cannot_link)
  rescued: set[int] = set()
  for _, idx, pos in candidates:
    if idx in rescued:
      continue
    target = ordered[pos][0]
    # 라이브 labels 검사 — 먼저(더 높은 유사도로) 구제된 상대가 있으면 그 클러스터는 차단된다
    if any(labels[partner] == target for partner in partners.get(idx, ())):
      continue
    labels[idx] = target
    rescued.add(idx)


def _margin_rescue_noise(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  candidates: Sequence[int],
  config: ClusterConfig,
) -> None:
  """2위 군집 대비 여유(margin)가 큰 잔여 노이즈를 top1 군집에 편입한다 (labels 제자리 수정).

  ④ 절대 임계 구제가 놓치는 하드 포즈(옆얼굴·역광·모션) 전용의 2차 구제다 — 근거·실측·한계는
  ClusterConfig.margin_rescue_floor 주석, 후보 선정(축출 강등분 포함)의 근거는 _recluster_core
  주석 참조. ④와 달리 top1 군집만 후보로 삼는다: top1이
  cannot-link로 차단됐을 때 top2로 낙착하면 "2위와는 멀어서 믿는다"는 게이트의 전제 자체가
  깨진다. 군집이 1개뿐이면 top2가 없어 여유를 정의할 수 없으므로 보수적으로 전체를 건너뛴다
  (단일 군집 이벤트에서 무차별 편입 방지). 처리 순서는 ④와 같은 전역 top1 유사도 내림차순 —
  cannot-link 경합에서 더 나은 매치가 항상 먼저 배정되어 결과가 결정적이다.
  """
  if not candidates:
    return
  ordered = _cluster_groups(labels)
  if len(ordered) < 2:
    return
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  similarities = embeddings[np.asarray(candidates)] @ centroids.T  # (후보 수, 클러스터 수)
  ranked: list[tuple[float, int, int]] = []
  for row, idx in enumerate(candidates):
    order = np.argsort(similarities[row])
    top1, top2 = float(similarities[row, order[-1]]), float(similarities[row, order[-2]])
    # top2 ≤ 0(경쟁자 없음)은 배율 조건이 자동 충족 — max(top2, 0)으로 나눗셈 없이 판정한다
    if top1 >= config.margin_rescue_floor and top1 >= config.margin_rescue_ratio * max(top2, 0.0):
      ranked.append((-top1, idx, int(order[-1])))
  partners = _cannot_link_partners(cannot_link)
  for _, idx, pos in sorted(ranked):
    target = ordered[pos][0]
    # 라이브 labels 검사 — 먼저 구제된 cannot-link 상대가 있으면 편입하지 않는다 (top2 낙착 없음)
    if any(labels[partner] == target for partner in partners.get(idx, ())):
      continue
    labels[idx] = target


def _evict_ambiguous(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  protected: set[int],
  config: ClusterConfig,
) -> tuple[int, ...]:
  """저신뢰 멤버를 클러스터에서 분리한다 (labels는 노이즈로 수정, ambiguous 인덱스만 반환).

  자신 없는 배정을 인물 앨범에 넣지 않는다 (feature-spec §7, TBD #3의 기본 정책):
  - 자기 클러스터 유사도가 바닥(min_membership_similarity) 미만 → 어디에도 속하지 않는 얼굴이므로
    노이즈로 강등한다 (반환 목록에는 없음 — noise_indices로 집계된다).
  - 바닥은 넘었지만 회색지대(< evict_gray_ceiling)인 멤버는 face-pair 증거로 재확인한다 (ADR 020):
    클러스터 내 최강 쌍이 evict_facepair_floor 미만이면 클러스터의 누구와도 동일인 증거가 없는
    남남 부착이므로 노이즈로 강등한다. centroid는 회색지대에서 동일인/남남이 섞여(전역 바닥 상향
    불가) 판별 신호가 개별 쌍에만 남는다 — ADR-016(파편병합 face-pair 게이트)과 같은 원리.
  - 2위 클러스터와의 유사도 마진이 min_membership_margin 미만 → 두 인물 사이의 애매한 얼굴이므로
    ambiguous로 분리해 반환한다.
  자기 클러스터 유사도는 leave-one-out centroid(자기를 뺀 평균) 기준이다 — 자기가 포함된
  centroid는 유사도를 부풀려 경계 얼굴이 마진 검사를 통과해 버린다. 두 가지 예외:
  - 사용자 제약에 직접 걸린 얼굴(must-link 컴포넌트, cannot-link 당사자 — 호출자가 protected로 전달)과
    단독 멤버 클러스터(자기가 곧 centroid)는 빼지 않는다.
  - cannot-link로 연결된 클러스터 쌍은 마진 비교에서 서로 제외한다 — 사용자가 갈라둔 동일 인물
    양쪽에 가까운 것은 당연하므로, 분리 유지가 애매함으로 오판되면 split된 앨범이 전부 비게 된다.
  평가는 시작 시점 멤버십 스냅샷으로 일괄 수행해 축출 순서에 결과가 의존하지 않는다.
  """
  ordered = _cluster_groups(labels)
  if not ordered:
    return ()
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  position_of = {label: pos for pos, (label, _) in enumerate(ordered)}
  count = len(ordered)
  linked = np.zeros((count, count), dtype=bool)
  for a, b in cannot_link:
    label_a, label_b = int(labels[a]), int(labels[b])
    if label_a != _NOISE and label_b != _NOISE and label_a != label_b:
      pos_a, pos_b = position_of[label_a], position_of[label_b]
      linked[pos_a, pos_b] = linked[pos_b, pos_a] = True

  all_sims = embeddings @ centroids.T  # (N, 클러스터 수) — 멤버별 matmul 대신 1회 BLAS 호출 (N=8천에서 ~3배 차이)
  gray_gate = config.evict_gray_ceiling > 0 and config.evict_facepair_floor > 0
  demoted_noise: list[int] = []
  ambiguous: list[int] = []
  for pos, (_, members) in enumerate(ordered):
    if len(members) < 2:
      continue
    member_arr = np.asarray(members)
    loo_sims = _loo_similarities(embeddings, members)
    unlinked = [q for q in range(count) if q != pos and not linked[pos, q]]
    others_max = all_sims[member_arr][:, unlinked].max(axis=1) if unlinked else None
    pair_sims = None
    if gray_gate and bool((loo_sims < config.evict_gray_ceiling).any()):
      pair_sims = embeddings[member_arr] @ embeddings[member_arr].T  # 회색지대 멤버가 있을 때만 계산
    for row, idx in enumerate(members):
      if idx in protected:
        continue
      sim_own = float(loo_sims[row])
      if sim_own < config.min_membership_similarity:
        demoted_noise.append(idx)
      elif (
        pair_sims is not None
        and sim_own < config.evict_gray_ceiling
        and float(np.delete(pair_sims[row], row).max()) < config.evict_facepair_floor
      ):
        demoted_noise.append(idx)  # 회색지대 + 동일인 쌍 증거 부재 = 남남 부착 (ADR 020)
      elif others_max is not None and sim_own - float(others_max[row]) < config.min_membership_margin:
        ambiguous.append(idx)
  for idx in demoted_noise + ambiguous:
    labels[idx] = _NOISE
  return tuple(sorted(ambiguous))


def _match_cluster_ids(
  new_clusters: list[tuple[int, list[int]]],
  previous_cluster_ids: Sequence[str | None],
  min_match_jaccard: float,
) -> tuple[dict[int, str], tuple[str, ...]]:
  """신규 파티션 ↔ 기존 클러스터를 Jaccard 내림차순 greedy 1:1 매칭한다 (feature-spec §4 ④).

  스펙의 'overlap 최대 매칭(Jaccard / 헝가리안)' 중 greedy Jaccard를 채택 — 가장 강한 겹침이
  그 번호를 가져가는 규칙이 결정적·설명 가능하고, numpy 전용 원칙(scipy 헝가리안 배제)과 맞다.
  동률은 (교집합 크기 내림차순 → 기존 id 등장 순 → 신규 클러스터 순)으로 고정한다.
  반환: (신규 클러스터 위치 → 승계한 cluster_id, 승계되지 못해 은퇴하는 기존 id들).
  """
  previous_members: dict[str, set[int]] = {}
  previous_order: list[str] = []  # 첫 등장 순서 — 은퇴 목록과 동률 처리의 결정성
  for idx, previous_id in enumerate(previous_cluster_ids):
    if previous_id is None:
      continue
    if previous_id not in previous_members:
      previous_members[previous_id] = set()
      previous_order.append(previous_id)
    previous_members[previous_id].add(idx)

  candidates: list[tuple[float, int, int, int]] = []  # (-jaccard, -교집합, 기존 순번, 신규 순번)
  for new_pos, (_, members) in enumerate(new_clusters):
    member_set = set(members)
    for prev_pos, previous_id in enumerate(previous_order):
      intersection = len(member_set & previous_members[previous_id])
      if intersection == 0:
        continue
      union = len(member_set) + len(previous_members[previous_id]) - intersection
      jaccard = intersection / union
      if jaccard < min_match_jaccard:
        continue
      candidates.append((-jaccard, -intersection, prev_pos, new_pos))

  matched: dict[int, str] = {}
  used_previous: set[int] = set()
  for _, _, prev_pos, new_pos in sorted(candidates):
    if new_pos in matched or prev_pos in used_previous:
      continue
    matched[new_pos] = previous_order[prev_pos]
    used_previous.add(prev_pos)

  retired = tuple(pid for pos, pid in enumerate(previous_order) if pos not in used_previous)
  return matched, retired


def _duplicate_row_groups(emb: np.ndarray, constraints: Constraints, config: ClusterConfig) -> list[list[int]]:
  """근중복 행 그룹(쌍 유사도 ≥ duplicate_collapse_similarity, 전이적 연결)을 찾는다 (⓪, ADR-029).

  사람 결정 우선: 그룹 안에 사용자 cannot-link 쌍이 걸려 있으면 임베딩상 복제로 보여도 그 그룹은
  통째로 접지 않는다 — 사람이 타인이라 지목한 두 행을 붕괴가 한 행으로 삼키면 안 된다.
  """
  threshold = config.duplicate_collapse_similarity
  n = emb.shape[0]
  if threshold <= 0.0 or n < 2:
    return []
  gram = emb @ emb.T
  pair_rows, pair_cols = np.nonzero(np.triu(gram >= threshold, k=1))
  if pair_rows.size == 0:
    return []
  uf = _UnionFind(n)
  for i, j in zip(pair_rows.tolist(), pair_cols.tolist()):
    uf.union(i, j)
  members: dict[int, list[int]] = {}
  for row in range(n):
    members.setdefault(uf.find(row), []).append(row)
  groups = sorted(rows for rows in members.values() if len(rows) >= 2)
  user_cannot = {(min(pair), max(pair)) for pair in constraints.cannot_link}
  kept_groups = []
  for group in groups:
    group_set = set(group)
    if any(i in group_set and j in group_set for i, j in user_cannot):
      continue
    kept_groups.append(group)
  return kept_groups


def _collapse_and_recluster(
  emb: np.ndarray,
  previous_cluster_ids: Sequence[str | None],
  resolved_constraints: Constraints,
  resolved_config: ClusterConfig,
  factory: Callable[[], str],
  groups: list[list[int]],
) -> ReclusterResult | None:
  """근중복 그룹을 대표 1행으로 접어 재군집하고 결과 인덱스를 원 행으로 펼친다 (⓪, ADR-029).

  접기가 사용자 must/cannot-link에 새 모순을 만들면(복제 행들이 상반된 보정에 걸린 극단 케이스)
  None을 반환한다 — 호출자는 붕괴를 포기하고 비붕괴 경로로 폴백한다 (사람 결정 우선).
  """
  rep_of: dict[int, int] = {}
  rows_of_rep: dict[int, list[int]] = {}
  for group in groups:
    rep = group[0]
    rows_of_rep[rep] = group
    for row in group:
      rep_of[row] = rep
  kept = [row for row in range(emb.shape[0]) if rep_of.get(row, row) == row]
  pos_of = {row: pos for pos, row in enumerate(kept)}

  # 대표 행의 직전 배정은 그룹에서 처음 등장하는 non-None id — 대표가 신규 행이어도 복제/유령 행이
  # 갖고 있던 앨범 번호 연속성(Jaccard 승계 투표)을 잃지 않는다.
  collapsed_previous: list[str | None] = []
  for row in kept:
    rows = rows_of_rep.get(row, [row])
    collapsed_previous.append(
      next((previous_cluster_ids[r] for r in rows if previous_cluster_ids[r] is not None), None)
    )

  def remap(pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    mapped = []
    for i, j in pairs:
      a, b = pos_of[rep_of.get(i, i)], pos_of[rep_of.get(j, j)]
      if a != b:  # 그룹 내부 쌍은 자기 자신 제약이 되므로 탈락 (cannot 내부 쌍은 그룹 선별에서 이미 제외)
        mapped.append((a, b))
    return tuple(dict.fromkeys(mapped))

  collapsed_constraints = Constraints(
    must_link=remap(resolved_constraints.must_link),
    cannot_link=remap(resolved_constraints.cannot_link),
    auto_cannot_link=remap(resolved_constraints.auto_cannot_link),
  )
  try:
    # 붕괴가 만든 모순의 사전 탐지 — 원 제약이 애초에 모순이면 폴백한 비붕괴 경로가 같은 계약대로 raise한다
    _must_link_components(len(kept), collapsed_constraints)
  except ValueError:
    return None

  core = _recluster_core(emb[kept], collapsed_previous, collapsed_constraints, resolved_config, factory)

  def expand(indices: Sequence[int]) -> tuple[int, ...]:
    rows: list[int] = []
    for pos in indices:
      row = kept[pos]
      rows.extend(rows_of_rep.get(row, [row]))
    return tuple(sorted(rows))

  clusters = []
  for cluster in core.clusters:
    member_rows: list[tuple[int, float]] = []
    for pos, similarity in zip(cluster.member_indices, cluster.membership_similarities):
      row = kept[pos]
      member_rows.extend((r, similarity) for r in rows_of_rep.get(row, [row]))  # 복제 행은 대표의 LOO를 공유
    member_rows.sort()
    clusters.append(
      PersonCluster(
        cluster_id=cluster.cluster_id,
        is_new=cluster.is_new,
        member_indices=tuple(r for r, _ in member_rows),
        membership_similarities=tuple(s for _, s in member_rows),
        # 접힌 멤버 기준 평균 그대로 — 복제를 다시 넣어 평균하면 재업로드된 사진 쪽으로 대표벡터가 쏠린다
        centroid=cluster.centroid,
      )
    )
  # 고아 근중복 그룹 승격 — 재군집이 클러스터를 하나도 못 만든 이벤트(예: 근중복 묶음이 사실상 전부)
  # 에서는 접힌 대표가 홀로 남아 min_cluster_size를 못 채운다. 비붕괴 경로가 이런 이벤트에 만들던
  # 앨범(자가검증 (d) 버스트 계약)을 보존하기 위해, 원 행 수가 min_cluster_size 이상인 노이즈 그룹만
  # 클러스터로 승격한다. 클러스터가 있는 이벤트에서는 승격하지 않는다 — 흡수에 실패한 재업로드 쌍을
  # 승격하면 쌍 앨범(P0 증상)이 되살아난다. 그런 행은 노이즈(uncertain)로 남는 것이 옳다
  # (재업로드·유령 행 = 같은 얼굴의 1회 등장, single_appearance와 같은 제품 의미).
  noise_positions: list[int] = list(core.noise_indices)
  if not core.clusters:
    promoted: set[int] = set()
    used_ids: set[str] = set()
    for pos in noise_positions:
      rows = rows_of_rep.get(kept[pos], [kept[pos]])
      if len(rows) < resolved_config.min_cluster_size:
        continue
      previous_id = collapsed_previous[pos]
      inherited_id = previous_id if previous_id is not None and previous_id not in used_ids else None
      if inherited_id is not None:
        used_ids.add(inherited_id)
      centroid = _normalized_mean(emb, rows)
      centroid.flags.writeable = False
      clusters.append(
        PersonCluster(
          cluster_id=inherited_id if inherited_id is not None else factory(),
          is_new=inherited_id is None,
          member_indices=tuple(rows),
          membership_similarities=tuple(float(s) for s in _loo_similarities(emb, rows)),
          centroid=centroid,
        )
      )
      promoted.add(pos)
    noise_positions = [pos for pos in noise_positions if pos not in promoted]
  clusters.sort(key=lambda cluster: cluster.member_indices[0])  # 결과 계약: 최소 멤버 인덱스 오름차순

  # 은퇴 목록은 원 행 기준으로 재계산 — 접기로 사라진 행만 갖고 있던 기존 id도 승계 실패면 은퇴에 들어가야
  # 하류(Spring)가 유령 인물을 잡고 있지 않는다. 순서는 비붕괴 경로와 동일한 첫 등장 순.
  inherited = {cluster.cluster_id for cluster in clusters if not cluster.is_new}
  retired = tuple(dict.fromkeys(pid for pid in previous_cluster_ids if pid is not None and pid not in inherited))
  return ReclusterResult(
    clusters=tuple(clusters),
    noise_indices=expand(noise_positions),
    ambiguous_indices=expand(core.ambiguous_indices),
    retired_cluster_ids=retired,
  )


def recluster(
  embeddings: np.ndarray,
  previous_cluster_ids: Sequence[str | None],
  constraints: Constraints | None = None,
  config: ClusterConfig | None = None,
  new_id_factory: Callable[[], str] | None = None,
) -> ReclusterResult:
  """event 전체 임베딩을 재군집하고 기존 cluster_id를 재조정한다 (feature-spec §4 ③④⑤).

  재군집 전 근중복 행 붕괴(⓪, ADR-029)로 재업로드·유령 행 오염을 접은 뒤, 재군집과 결정적 후처리를
  순서대로 적용한다: 보정 강제(must→cannot-link) → 파편 병합 → 노이즈 구제 → 저신뢰 ambiguous 분리 →
  2차 파편 병합 → ID 재조정 → 대표벡터 (모듈 독스트링 ⓪~⑧). 결과 인덱스는 항상 원 행 기준이다.

  Args:
    embeddings: shape (N, EMBED_DIM) — event 전체(기존+신규) 임베딩. L2 정규화 단위벡터 전제.
    previous_cluster_ids: 길이 N — 각 행의 직전 클러스터 배정 (신규·직전 노이즈는 None).
    constraints: 사용자 보정 제약. 모순 셋은 ValueError.
    config: HDBSCAN·후처리·매칭 파라미터 (기본: PoC 검증 레시피 + 보수적 후처리 임계).
    new_id_factory: 신규 cluster_id 발급자 (기본 uuid4) — 테스트에서 결정적 주입용.

  같은 입력(과 같은 factory)에 대해 결과는 항상 동일하다(결정적).
  """
  resolved_config = config if config is not None else ClusterConfig()
  resolved_constraints = constraints if constraints is not None else Constraints()
  factory = new_id_factory if new_id_factory is not None else (lambda: str(uuid.uuid4()))

  emb = np.asarray(embeddings)
  if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
    raise ValueError(f"embeddings는 shape (N, {EMBED_DIM})이어야 합니다. 받은 shape: {emb.shape}")
  if emb.size and not np.isfinite(emb).all():
    # 비유한 벡터는 cosine 거리가 정의되지 않아 군집 전체를 오염시킨다 — embed 단계가 None으로
    # 걸러 보냈어야 하는 값이므로 프로그래밍 오류로 거부한다 (embed._preprocess와 동일 철학).
    raise ValueError("embeddings에 비유한값(NaN/inf)이 있습니다. embed 단계는 퇴화 임베딩을 걸러야 합니다.")
  if emb.size:
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
      # HDBSCAN cosine 경로는 내부 정규화하지만 병합·구제·저신뢰 후처리는 단위벡터 전제의 생 내적을
      # 코사인으로 쓴다 — 비정규 입력은 모든 유사도 임계를 조용히 우회하는 것이 리뷰에서 재현됐으므로
      # 거부한다 (embed는 항상 L2 정규화 출력, 저장소 왕복 오차는 atol로 흡수).
      worst = float(norms[int(np.argmax(np.abs(norms - 1.0)))])
      raise ValueError(f"embeddings는 L2 정규화 단위벡터여야 합니다. 받은 norm 예: {worst:.4f}")
  n = emb.shape[0]
  if len(previous_cluster_ids) != n:
    raise ValueError(
      f"previous_cluster_ids 길이는 임베딩 수와 같아야 합니다. 받은 길이: {len(previous_cluster_ids)}, 임베딩 수: {n}"
    )
  _validate_pairs(resolved_constraints.must_link, n, "must-link")
  _validate_pairs(resolved_constraints.cannot_link, n, "cannot-link")
  _validate_pairs(resolved_constraints.auto_cannot_link, n, "auto-cannot-link")

  # ⓪ 근중복 행 붕괴 (ADR-029) — HDBSCAN이 거리 0의 복제 쌍을 실제 인물 덩어리보다 우선 선택해
  # 앨범이 쌍 단위로 와해되는 재업로드·유령 행 오염(실 event 8)을 재군집 전에 차단한다.
  groups = _duplicate_row_groups(emb, resolved_constraints, resolved_config)
  if groups:
    collapsed = _collapse_and_recluster(
      emb, previous_cluster_ids, resolved_constraints, resolved_config, factory, groups
    )
    if collapsed is not None:
      return collapsed
  return _recluster_core(emb, previous_cluster_ids, resolved_constraints, resolved_config, factory)


def _recluster_core(
  emb: np.ndarray,
  previous_cluster_ids: Sequence[str | None],
  resolved_constraints: Constraints,
  resolved_config: ClusterConfig,
  factory: Callable[[], str],
) -> ReclusterResult:
  """검증과 근중복 붕괴(⓪)가 끝난 입력의 재군집 본체 (모듈 독스트링 ①~⑧)."""
  n = emb.shape[0]
  comp_of, components = _must_link_components(n, resolved_constraints)
  # 자동 제약 결합 (Constraints 독스트링·ADR-011): 사람 must-link와 모순되는 자동 쌍은 탈락시키고,
  # 나머지는 병합·분리·구제에서 사람 cannot-link와 동일하게 참여한다. 축출(보호·마진 제외)은
  # 사람 cannot-link만 쓴다 — 아래에서 resolved_constraints.cannot_link를 직접 참조하는 이유.
  blocking_cannot = tuple(
    dict.fromkeys(
      (
        *resolved_constraints.cannot_link,
        *(pair for pair in resolved_constraints.auto_cannot_link if comp_of[pair[0]] != comp_of[pair[1]]),
      )
    )
  )

  # ③ 전체 재군집 — 표본이 min_samples·min_cluster_size 미만이면 밀도 군집이 정의되지 않으므로
  # 전원 노이즈로 두고 제약 후처리만 적용한다 (이식본은 min_samples > N에서 raise하므로 사전 분기).
  if n >= max(resolved_config.min_cluster_size, resolved_config.min_samples):
    labels = HDBSCAN(
      min_cluster_size=resolved_config.min_cluster_size,
      min_samples=resolved_config.min_samples,
      metric="cosine",
      cluster_selection_epsilon=resolved_config.cluster_selection_epsilon,
    ).fit_predict(emb)
    labels = np.asarray(labels, dtype=np.int64)
    # 균질 blob 퇴화(클러스터 0개) 교정 — 제약 강제 전에 승격해야 cannot-link가 승격된 라벨을 분리할 수 있다
    _promote_single_blob(labels, emb, resolved_config)
  else:
    labels = np.full(n, _NOISE, dtype=np.int64)

  # 사용자 보정 강제 — must-link(병합)를 먼저 적용해야 cannot-link(분리)가 최종 상태에서 위반을 본다
  next_label = int(labels.max()) + 1 if n else 0
  next_label = _enforce_must_link(labels, components, next_label, blocking_cannot)
  _enforce_cannot_link(labels, comp_of, components, blocking_cannot, emb, next_label)

  # 파편 병합 → 노이즈 구제 → 저신뢰 분리 (순서 중요: 병합된 centroid 기준으로 구제하고,
  # 구제까지 끝난 최종 멤버십에서 저신뢰를 가려낸다. 병합을 구제 뒤로 옮기면 구제가 파편난
  # 작은 centroid 기준이 되어 손해라, 시점 차이는 이동이 아니라 아래 2차 병합으로 보완한다)
  _merge_fragments(
    labels,
    emb,
    blocking_cannot,
    resolved_config.merge_centroid_similarity,
    resolved_config.merge_facepair_floor,
    resolved_config.merge_component_linkage,
  )
  _rescue_noise(labels, emb, blocking_cannot, resolved_config.rescue_similarity)
  protected = {idx for members in components.values() if len(members) >= 2 for idx in members}
  # cannot-link 당사자도 보호 — split로 생긴 소형 클러스터는 구성상 내부 유사도가 낮을 수 있어,
  # 절대 바닥 축출이 클러스터를 통째로 비워 사용자 분리 결정과 cluster_id를 지우는 것이 리뷰에서
  # 재현됐다. 제약 당사자는 사람이 직접 지목한 얼굴이므로 어느 축출 경로로도 빼지 않는다.
  # 자동 제약(auto_cannot_link)은 여기 불참 — 사람이 지목한 얼굴이 아니고, 참여시키면 사실상 전
  # 얼굴이 보호되어 축출이 무력화된다 (Constraints 독스트링·ADR-011).
  protected.update(idx for pair in resolved_constraints.cannot_link for idx in pair)
  ambiguous_indices = _evict_ambiguous(labels, emb, resolved_constraints.cannot_link, protected, resolved_config)
  ambiguous_set = set(ambiguous_indices)
  if resolved_config.margin_rescue_floor > 0:
    # 후보 = 축출까지 끝난 최종 노이즈 전부(축출의 회색지대 강등분 포함) − ambiguous.
    # 강등분을 제외하지 않는 이유: HDBSCAN이 하드 포즈를 일단 클러스터에 붙였다가 회색지대 게이트가
    # 강등하는 경로가 흔해(옆얼굴은 개별 쌍도 전부 0.45 미만이라 ADR-020 증거를 못 만든다), 제외하면
    # 게이트가 사실상 죽는다(성인 9인 실측: 제외 시 구제 1건 vs 포함 시 3건·F1 +0.056). 회색지대
    # 축출(절대·쌍 증거 부재)과 margin(상대 여유)은 다른 증거축의 재심이라 실행 내 재순환도 없다 —
    # margin이 멤버십의 최종 발언이고 매 실행 같은 고정점에 수렴한다. ambiguous는 '2위 마진 부족'
    # 판정이라 margin과 같은 축에서 이미 기각된 것이므로 제외한다.
    margin_candidates = sorted(int(i) for i in np.flatnonzero(labels == _NOISE) if int(i) not in ambiguous_set)
    _margin_rescue_noise(labels, emb, blocking_cannot, margin_candidates, resolved_config)
  # 2차 파편 병합 (ADR-010) — 1차 병합은 구제·축출 전 centroid 스냅샷으로 판정하므로, 구제가 멤버를
  # 추가하면 최종 구성 기준으로는 임계를 넘는 파편 쌍이 남는다 (실 event 실측: 판정 시 0.688 →
  # 구제 후 0.705). 같은 임계·같은 cannot-link 가드를 최종 멤버십에서 한 번 더 적용한다.
  _merge_fragments(
    labels,
    emb,
    blocking_cannot,
    resolved_config.merge_centroid_similarity,
    resolved_config.merge_facepair_floor,
    resolved_config.merge_component_linkage,
  )

  # ID 재조정
  new_clusters = _cluster_groups(labels)
  matched, retired = _match_cluster_ids(new_clusters, previous_cluster_ids, resolved_config.min_match_jaccard)

  # 대표벡터 계산 + 결과 조립 (신규 id 발급은 출력 순서대로 — factory 주입 시 결정성 보장)
  clusters = []
  for new_pos, (_, members) in enumerate(new_clusters):
    inherited = matched.get(new_pos)
    centroid = _normalized_mean(emb, members)
    centroid.flags.writeable = False  # frozen dataclass 출력이 하류에서 변형되지 않도록 보호
    clusters.append(
      PersonCluster(
        cluster_id=inherited if inherited is not None else factory(),
        is_new=inherited is None,
        member_indices=tuple(members),
        membership_similarities=tuple(float(s) for s in _loo_similarities(emb, members)),
        centroid=centroid,
      )
    )
  noise_indices = tuple(int(idx) for idx in np.flatnonzero(labels == _NOISE) if int(idx) not in ambiguous_set)
  return ReclusterResult(
    clusters=tuple(clusters),
    noise_indices=noise_indices,
    ambiguous_indices=ambiguous_indices,
    retired_cluster_ids=retired,
  )


if __name__ == "__main__":
  import sys

  if sys.argv[1:]:
    # 이미지 CLI 모드 — SQS/S3 없이 파이프라인 파리티를 확인: 로컬 이미지들에서 검출→정렬→임베딩→재군집을
    # 실행해 인물 클러스터 구성을 출력한다 (최초 군집 시나리오 — previous_cluster_ids 전부 None).
    import time

    # detect/embed는 onnxruntime·huggingface_hub 임포트 체인을 끌고 오므로 CLI 확인 블록에서만 지연 import한다
    import cv2

    from app.pipeline.align import align_face
    from app.pipeline.detect import FaceDetector
    from app.pipeline.embed import FaceEmbedder

    detector = FaceDetector()
    embedder = FaceEmbedder()
    face_names: list[str] = []
    face_embeddings: list[np.ndarray] = []
    for path in sys.argv[1:]:
      image = cv2.imread(path)
      if image is None:
        print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
        continue
      detected = detector.detect(image)
      crops = [(i, align_face(image, face.landmarks)) for i, face in enumerate(detected)]
      valid = [(i, crop) for i, crop in crops if crop is not None]
      for (face_i, _), embedding in zip(valid, embedder.embed_batch([crop for _, crop in valid])):
        if embedding is None:
          continue
        face_names.append(f"{path}#face{face_i}")
        face_embeddings.append(embedding)
      print(f"{path}: {len(detected)} face(s), 임베딩 {len(face_embeddings)}개 누적")

    if not face_embeddings:
      print("클러스터링할 얼굴이 없습니다.")
      sys.exit(0)

    start = time.perf_counter()
    result = recluster(np.stack(face_embeddings), [None] * len(face_embeddings))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(
      f"\n{len(face_embeddings)}개 얼굴 → 클러스터 {len(result.clusters)}개, "
      f"노이즈 {len(result.noise_indices)}개, 저신뢰 {len(result.ambiguous_indices)}개 in {elapsed_ms:.1f} ms"
    )
    for cluster in result.clusters:
      print(f"  [{cluster.cluster_id}] is_new={cluster.is_new}, 멤버 {len(cluster.member_indices)}명")
      for idx in cluster.member_indices:
        print(f"    {face_names[idx]}")
    for idx in result.noise_indices:
      print(f"  노이즈: {face_names[idx]}")
    for idx in result.ambiguous_indices:
      print(f"  저신뢰(ambiguous): {face_names[idx]}")
    sys.exit(0)

  # 인자 없음: 모델·이미지 없이 합성 임베딩으로 연결 성분 부분 승격(ADR-008)을 자가 검증한다.
  # 쌍 유사도를 닫힌형식으로 제어한 벡터로 "소규모 단일 인물 이벤트 전원 uncertain" 퇴화와
  # 그 교정을 결정적으로 재현한다 (실측 대역: 동일 인물 쌍 0.46~0.70, 타인 ≲0.3).
  # TODO(CHMO-165): pytest 도입 시 tests/test_cluster.py로 승격
  import math

  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  def axis(i: int) -> np.ndarray:
    vector = np.zeros(EMBED_DIM, dtype=np.float64)
    vector[i] = 1.0
    return vector

  def spread_vectors(cosines: Sequence[float], base: np.ndarray, spread_axis0: int) -> np.ndarray:
    """멤버 i = c_i·base + √(1-c_i²)·(고유 직교 축) — 쌍 유사도가 정확히 c_i·c_j·(base_i·base_j)인 단위벡터.

    base 성분의 곱 구조라 MST가 star형이 되어 HDBSCAN condensed tree에 진짜 분할 지점이 없다 —
    실사진 소규모 단일 인물 이벤트의 "클러스터 0개" 퇴화를 임계값까지 통제하며 재현한다.
    """
    vectors = np.zeros((len(cosines), EMBED_DIM), dtype=np.float64)
    for row, c in enumerate(cosines):
      vectors[row] = c * base
      vectors[row, spread_axis0 + row] = math.sqrt(1.0 - c * c)
    return vectors.astype(np.float32)

  def hdbscan_labels(vectors: np.ndarray) -> np.ndarray:
    config = ClusterConfig()
    return np.asarray(
      HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="cosine",
        cluster_selection_epsilon=config.cluster_selection_epsilon,
      ).fit_predict(vectors),
      dtype=np.int64,
    )

  def raises_value_error(factory: Callable[[], object]) -> bool:
    try:
      factory()
    except ValueError:
      return True
    return False

  # (a) 실측 대역 단일 인물 5장 — 쌍 유사도 0.533~0.648 (전부 구 임계 0.7 미만)
  real_band = spread_vectors((0.82, 0.79, 0.76, 0.74, 0.72), axis(0), 10)
  check(
    "(a) 전제: 실측 대역 5장은 HDBSCAN 클러스터 0개(전원 노이즈) 퇴화를 밟는다",
    bool((hdbscan_labels(real_band) == _NOISE).all()),
  )
  result = recluster(real_band, [None] * 5)
  check(
    "(a) 실측 대역 단일 인물 5장 → 클러스터 1개에 전원 소속 (앨범 미생성 버그 교정)",
    len(result.clusters) == 1
    and result.clusters[0].member_indices == (0, 1, 2, 3, 4)
    and result.noise_indices == ()
    and result.ambiguous_indices == (),
  )

  # (b) 단일 인물 4장 + 낯선 행인 1장 — 교차 유사도 ≈0.13~0.15 (부분 승격: 행인만 노이즈 유지)
  person = spread_vectors((0.82, 0.79, 0.76, 0.73), axis(0), 10)
  stranger_base = 0.2 * axis(0) + math.sqrt(1.0 - 0.2**2) * axis(1)
  stranger = spread_vectors((0.9,), stranger_base, 20)
  mixed = np.vstack([person, stranger])
  check("(b) 전제: 인물 4장 + 행인도 HDBSCAN 전원 노이즈", bool((hdbscan_labels(mixed) == _NOISE).all()))
  result = recluster(mixed, [None] * 5)
  check(
    "(b) 인물 4장만 승격, 행인은 노이즈 유지 (전체 일괄 승격이면 불가능한 결과)",
    len(result.clusters) == 1 and result.clusters[0].member_indices == (0, 1, 2, 3) and result.noise_indices == (4,),
  )

  # (c) 두 인물이 전원 노이즈로 시작 — 교차 ≈0.26~0.32: 성분 2개가 각각 승격, 섞임 없음.
  # recluster e2e로는 재현 불가(두 밀집은 HDBSCAN이 정상 분할)라 헬퍼를 직접 검증한다.
  second_base = 0.4 * axis(0) + math.sqrt(1.0 - 0.4**2) * axis(1)
  two_people = np.vstack(
    [spread_vectors((0.9, 0.85, 0.8), axis(0), 10), spread_vectors((0.9, 0.85, 0.8), second_base, 20)]
  )
  labels = np.full(6, _NOISE, dtype=np.int64)
  _promote_single_blob(labels, two_people, ClusterConfig())
  check(
    "(c) 두 인물 성분 각각 승격 — 교차 ~0.3은 간선이 없어 섞이지 않음",
    labels.tolist() == [0, 0, 0, 1, 1, 1],
  )

  # (d) 근중복 버스트(쌍 ≥0.985) — 구 0.7 완전 연결 fast-path가 구제하던 케이스의 동작 보존
  theta = [math.radians(5.0 * k) for k in range(3)]
  burst = np.stack([math.cos(t) * axis(0) + math.sin(t) * axis(1) for t in theta]).astype(np.float32)
  check("(d) 전제: 근중복 버스트도 HDBSCAN 전원 노이즈", bool((hdbscan_labels(burst) == _NOISE).all()))
  result = recluster(burst, [None] * 3)
  check(
    "(d) 근중복 버스트 3장 → 클러스터 1개 (기존 승격 동작 회귀 없음)",
    len(result.clusters) == 1 and result.noise_indices == (),
  )

  # (e) 낯선 2인(유사도 0.1) → 승격 없음 / 동일 인물 2장(유사도 0.6) → 승격 (구제 범위 확대)
  strangers = np.vstack(
    [spread_vectors((1.0,), axis(0), 10), spread_vectors((1.0,), 0.1 * axis(0) + math.sqrt(0.99) * axis(1), 20)]
  )
  result = recluster(strangers, [None] * 2)
  check("(e) 낯선 2인은 승격되지 않음 (기존 가드 유지)", result.clusters == () and result.noise_indices == (0, 1))
  pair = spread_vectors((0.8, 0.75), axis(0), 10)  # 쌍 유사도 0.6
  result = recluster(pair, [None] * 2)
  check(
    "(e) 동일 인물 2장(0.6)은 승격 — 신규 구제 범위",
    len(result.clusters) == 1 and result.clusters[0].member_indices == (0, 1),
  )

  # (j)(k)의 합성 파편은 동일 벡터의 복제라 기본 설정에서는 ⓪ 근중복 붕괴(ADR-029)가 먼저 접는다 —
  # 검증 대상(2차 병합·같은사진 제약 기계)을 종전 기하 그대로 격리하기 위해 붕괴만 끈 설정을 쓴다.
  # 붕괴 경로 자체의 계약은 (n)이 검증한다.
  no_collapse = ClusterConfig(duplicate_collapse_similarity=0.0)

  # (j) 2차 파편 병합 (ADR-010) — 실 event 기하 재현: 파편 A·B가 1차 병합 시점엔 0.69로 임계(0.70)
  # 미달인데, 노이즈 w가 B로 구제되며 B centroid가 A 쪽으로 이동해 최종 구성으로는 임계를 넘는다.
  fragment_a = np.stack([axis(0), axis(0)]).astype(np.float32)  # centroid = e1
  b_dir = 0.69 * axis(0) + math.sqrt(1.0 - 0.69**2) * axis(1)  # A↔B = 0.69 < 0.70
  fragment_b = np.stack([b_dir, b_dir]).astype(np.float32)
  # w·b = 0.65, w·e1 = 0.57 — 구제 임계(0.60) 이상·argmax는 B, HDBSCAN 노이즈 대역(거리 0.35),
  # 저신뢰 축출 마진 0.08(> 0.05)로 축출을 면한다. 구제 후 B centroid의 A 유사도
  # = (2·0.69 + 0.57)/√(5 + 4·0.65) ≈ 0.707 ≥ 0.70 → 2차 병합 발동.
  w_e2 = (0.65 - 0.69 * 0.57) / math.sqrt(1.0 - 0.69**2)
  w = 0.57 * axis(0) + w_e2 * axis(1) + math.sqrt(1.0 - 0.57**2 - w_e2**2) * axis(2)
  rescued_split = np.vstack([fragment_a, fragment_b, w.astype(np.float32)])
  labels = hdbscan_labels(rescued_split)
  check(
    "(j) 전제: 파편 A·B는 분리 클러스터, w는 노이즈로 시작",
    len(set(labels[labels != _NOISE])) == 2 and labels[4] == _NOISE,
  )
  result = recluster(rescued_split, [None] * 5, config=no_collapse)
  check(
    "(j) 구제로 임계를 넘은 파편은 2차 병합으로 합류 — 전원 한 클러스터",
    len(result.clusters) == 1 and result.clusters[0].member_indices == (0, 1, 2, 3, 4),
  )

  # (k) 같은 사진 자동 cannot-link (ADR-011) — 행 구성: A 파편 [e1, e1] + B 파편 [b, b], A↔B = 0.72
  # (병합 임계 0.68 이상 → 제약 없으면 병합). auto 쌍 (0, 2)는 "0과 2가 같은 사진" = 타인 확정.
  k_b = 0.72 * axis(0) + math.sqrt(1.0 - 0.72**2) * axis(1)
  k_frags = np.vstack([np.stack([axis(0), axis(0)]), np.stack([k_b, k_b])]).astype(np.float32)
  result = recluster(k_frags, [None] * 4, config=no_collapse)
  check("(k) 전제: 0.72 파편 쌍은 제약 없으면 병합된다 (임계 0.68 하향 확인)", len(result.clusters) == 1)
  result = recluster(k_frags, [None] * 4, Constraints(auto_cannot_link=((0, 2),)), config=no_collapse)
  check(
    "(k) 같은 사진 얼굴이 걸친 파편 쌍은 유사도가 임계 이상이어도 병합 차단",
    len(result.clusters) == 2 and result.noise_indices == () and result.ambiguous_indices == (),
  )
  # 축출 비보호 대비 검증 — 행 구성: A [e1, e1, x] + B [b', b'], b'·e1 = 0.5.
  # x(=행 2)는 자기 클러스터 LOO 0.88, B와 0.838로 마진 0.042 < 0.05 → 원래 ambiguous 축출 대상.
  # 같은 기하에서 사람 cannot-link (2,3)은 당사자 보호 + 마진 비교 제외로 x를 지키지만,
  # auto 쌍 (2,3)은 어느 특혜도 없어 축출이 정상 동작해야 한다 (event-17 무의미 클러스터 회귀).
  k_b2 = 0.5 * axis(0) + math.sqrt(1.0 - 0.5**2) * axis(1)
  k_x = 0.88 * axis(0) + 0.46 * axis(1) + math.sqrt(1.0 - 0.88**2 - 0.46**2) * axis(2)
  k_mixed = np.vstack([np.stack([axis(0), axis(0)]), k_x[None, :], np.stack([k_b2, k_b2])]).astype(np.float32)
  result = recluster(k_mixed, [None] * 5, Constraints(cannot_link=((2, 3),)), config=no_collapse)
  check(
    "(k) 사람 cannot-link 당사자 x는 축출 보호로 유지 (기존 동작)",
    {c.member_indices for c in result.clusters} == {(0, 1, 2), (3, 4)} and result.ambiguous_indices == (),
  )
  result = recluster(k_mixed, [None] * 5, Constraints(auto_cannot_link=((2, 3),)), config=no_collapse)
  check(
    "(k) 같은 기하에서 auto 쌍의 x는 보호 없이 정상 축출 (축출 무력화 회귀 방지)",
    {c.member_indices for c in result.clusters} == {(0, 1), (3, 4)} and result.ambiguous_indices == (2,),
  )

  # (l) 파편병합 face-level 응집 게이트 (ADR-016) — centroid는 평균이라 서로 다른 아이도 뭉뚱그려
  # 임계를 넘기지만(포즈 노이즈 상쇄), 개별 얼굴 쌍엔 판별 신호가 남는다. 두 파편이 base(=아이 공통
  # 영역) 성분만 공유하고 나머지는 직교 축이면 cross face-pair 평균 = α²인데 centroid 유사도는
  # 2α²/(α²+1)로 더 높다 — event 35에서 실측한 갭(centroid 0.6·face 0.4)의 합성 재현.
  diff_a = math.sqrt(0.40)  # 다른 아이: cross 얼굴평균 0.40, centroid 0.571 (> 0.55)
  diff_b = math.sqrt(1.0 - 0.40)
  diff_kids = np.stack(
    [
      diff_a * axis(0) + diff_b * axis(1),
      diff_a * axis(0) + diff_b * axis(2),
      diff_a * axis(0) + diff_b * axis(3),
      diff_a * axis(0) + diff_b * axis(4),
    ]
  ).astype(np.float32)
  labels_off = np.array([0, 0, 1, 1], dtype=np.int64)
  _merge_fragments(labels_off, diff_kids, (), 0.55, 0.0)
  check("(l) 전제: floor=0이면 centroid 0.571로 서로 다른 아이 파편도 병합 (기존 동작 보존)", len(set(labels_off)) == 1)
  labels_on = np.array([0, 0, 1, 1], dtype=np.int64)
  _merge_fragments(labels_on, diff_kids, (), 0.55, 0.55)
  check("(l) face floor 0.55는 cross 얼굴평균 0.40인 다른 아이 파편의 병합을 차단", len(set(labels_on)) == 2)
  same_a = math.sqrt(0.65)  # 같은 인물: cross 얼굴평균 0.65, centroid 0.788
  same_b = math.sqrt(1.0 - 0.65)
  same_person = np.stack(
    [
      same_a * axis(0) + same_b * axis(1),
      same_a * axis(0) + same_b * axis(2),
      same_a * axis(0) + same_b * axis(3),
      same_a * axis(0) + same_b * axis(4),
    ]
  ).astype(np.float32)
  labels_same = np.array([0, 0, 1, 1], dtype=np.int64)
  _merge_fragments(labels_same, same_person, (), 0.55, 0.55)
  check(
    "(l) 같은 인물 파편(cross 얼굴평균 0.65)은 face floor 0.55를 통과해 병합 유지 — 성인 무회귀",
    len(set(labels_same)) == 1,
  )

  # (m) 파편병합 컴포넌트 전체 재평가 (ADR-024) — 구 완전 연결은 병합 전 파편 '스냅샷 쌍'을 검사해
  # 2얼굴 파편의 노이즈 centroid 쌍에 걸린다. event 90 기하의 합성 재현: 주 파편 M(4)·소형 파편
  # P(2)·Q(2)가 같은 인물인데 P·Q 서로만 게이트를 살짝 미달(실측 0.508/0.458), 컴포넌트 전체
  # (M∪P vs Q)로는 통과. base를 e0 기준 ±θ로 벌려 M-P·M-Q 통과 / P-Q 미달을 닫힌형식으로 만든다.
  def merge_gate_stats(f1: np.ndarray, f2: np.ndarray) -> tuple[float, float]:
    c1, c2 = f1.mean(axis=0), f2.mean(axis=0)
    centroid_sim = float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2)))
    return centroid_sim, float((f1 @ f2.T).mean())

  def three_frags(cos_t: float) -> np.ndarray:
    sin_t = math.sqrt(1.0 - cos_t * cos_t)
    return np.vstack(
      [
        spread_vectors((0.9, 0.9, 0.9, 0.9), axis(0), 10),
        spread_vectors((0.9, 0.9), cos_t * axis(0) + sin_t * axis(1), 20),
        spread_vectors((0.9, 0.9), cos_t * axis(0) - sin_t * axis(1), 30),
      ]
    ).astype(np.float32)

  def run_merge(embs: np.ndarray, component: bool) -> np.ndarray:
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    _merge_fragments(labels, embs, (), 0.55, 0.475, component)
    return labels

  heal = three_frags(0.88)  # M-P·M-Q 0.809/0.713 통과, P-Q 0.491/0.445 미달, M∪P vs Q 전체는 통과
  mp = merge_gate_stats(heal[0:4], heal[4:6])
  pq = merge_gate_stats(heal[4:6], heal[6:8])
  mq_comp = merge_gate_stats(heal[0:6], heal[6:8])
  check(
    "(m) 전제: M-P 게이트 통과 + P-Q 스냅샷 쌍 미달 + M∪P vs Q 컴포넌트 전체 통과 기하",
    mp[0] >= 0.55 and mp[1] >= 0.475 and pq[0] < 0.55 and pq[1] < 0.475 and mq_comp[0] >= 0.55 and mq_comp[1] >= 0.475,
  )
  check(
    "(m) 전제: 구 완전 연결은 P만 합류, Q는 P-Q 스냅샷 쌍에 걸려 잔류 (event 90 분리 재현)",
    len(set(run_merge(heal, False).tolist())) == 2,
  )
  check(
    "(m) 컴포넌트 전체 재평가는 같은 인물 파편 M·P·Q 전원 병합 (event 90 치유)",
    len(set(run_merge(heal, True).tolist())) == 1,
  )
  # 다리(bridge) 차단 유지 — 완전 연결을 없애는 대신 남남 얼굴 쌍이 컴포넌트 전체 face-pair 평균을
  # 끌어내리는 것이 융합 차단을 대행한다: 중간자 M이 남남 P·Q 둘 다와 게이트를 통과해도(cosθ=0.75)
  # M∪P vs Q의 전체 face평균(0.44)이 바닥(0.475)에 걸려 Q가 합류하지 못한다.
  bridge = three_frags(0.75)
  bp = merge_gate_stats(bridge[0:4], bridge[4:6])
  pq_bridge = merge_gate_stats(bridge[4:6], bridge[6:8])
  labels_bridge = run_merge(bridge, True)
  check(
    "(m) 다리 기하 전제: M-P·M-Q 게이트 통과 + P·Q는 남남 수준(face평균 ~0.10)",
    bp[0] >= 0.55 and bp[1] >= 0.475 and pq_bridge[1] < 0.2,
  )
  check(
    "(m) 컴포넌트 재평가도 다리 융합 차단 — M∪P는 성립하되 남남 Q는 같은 앨범 불가",
    labels_bridge[0] == labels_bridge[4] and labels_bridge[4] != labels_bridge[6],
  )

  # (f) 체이닝 차단 — A~B 0.5, B~C 0.5 간선으로 한 성분이지만 A~C가 0.3(floor 미만)이라
  # 두 남남(A·C)은 절대 같은 앨범에 남지 않는다. peel이 유사도 합 최소인 A를 떼고 나머지만 승격.
  chained = spread_vectors((0.548, 0.913, 0.548), axis(0), 10)  # 쌍: 0.500, 0.500, 0.300
  labels = np.full(3, _NOISE, dtype=np.int64)
  _promote_single_blob(labels, chained, ClusterConfig())
  check(
    "(f) 체이닝 남남(A~C 0.3)은 같은 앨범 불가 — peel이 A를 떼고 B·C만 승격",
    labels[0] == _NOISE and labels[1] == labels[2] != _NOISE,
  )

  def planar(degrees: Sequence[float], ax0: int, ax1: int) -> np.ndarray:
    """평면 위 각도로 배치한 단위벡터 — 쌍 유사도 = cos(각도차). peel 테스트용 비(非)곱 구조."""
    vectors = np.zeros((len(degrees), EMBED_DIM), dtype=np.float64)
    for row, deg in enumerate(degrees):
      t = math.radians(deg)
      vectors[row] = math.cos(t) * axis(ax0) + math.sin(t) * axis(ax1)
    return vectors.astype(np.float32)

  # (h) peel 구제 — 한 극단적 포즈(78°)가 floor를 깨는 성분: 통째 기각 대신 그 한 장만 떼고 핵심 3장 승격
  # (실사진 karina 5장에서 한 쌍 0.394가 성분 전체를 무너뜨리던 문제의 합성 재현). 직접 호출로 peel만 격리.
  poses = planar((0.0, 25.0, 50.0, 78.0), 0, 1)  # 0~2는 서로 ≥0.64, 3은 0과 0.21로 floor 미달
  labels = np.full(4, _NOISE, dtype=np.int64)
  _promote_single_blob(labels, poses, ClusterConfig())
  check(
    "(h) peel — 극단 포즈 1장만 노이즈로 떼고 핵심 3장은 한 앨범으로 승격",
    labels.tolist() == [0, 0, 0, _NOISE],
  )

  # (i) reassign 동률 회귀 — must-link {이동 얼굴, 목적지 대표}의 1:1 동률에서 cannot-link 금지 라벨(출처)을
  # 후보에서 제외해야 목적지 대표가 출처로 끌려나와 목적지 앨범이 쪼개지지 않는다
  # (docs/reviews/2026-07-10-reassign-mustlink-tiebreak.md — f1은 B 사람인데 A에 더 닮은 오분류 사진의 이동).
  # spread_vectors는 곱 구조라 이 기하(f1↔A > f1↔B, A·B 상호 0.3)를 못 만들어 목표 Gram을 고유분해한다.
  reassign_gram = np.eye(5)
  for j in (1, 2):
    reassign_gram[0, j] = reassign_gram[j, 0] = 0.55  # f1 ↔ A 사람 (오분류의 원인)
  for j in (3, 4):
    reassign_gram[0, j] = reassign_gram[j, 0] = 0.45  # f1 ↔ B 사람 (진짜 소속)
  reassign_gram[1, 2] = reassign_gram[2, 1] = 0.65
  reassign_gram[3, 4] = reassign_gram[4, 3] = 0.65
  for i in (1, 2):
    for j in (3, 4):
      reassign_gram[i, j] = reassign_gram[j, i] = 0.30
  eigvals, eigvecs = np.linalg.eigh(reassign_gram)
  reassign_emb = np.zeros((5, EMBED_DIM))
  reassign_emb[:, :5] = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
  reassign_emb = (reassign_emb / np.linalg.norm(reassign_emb, axis=1, keepdims=True)).astype(np.float32)
  result = recluster(
    reassign_emb,
    ["a", "a", "a", "b", "b"],
    constraints=Constraints(must_link=((0, 3),), cannot_link=((0, 1),)),
  )
  albums = {c.cluster_id: c.member_indices for c in result.clusters}
  check(
    "(i) reassign 동률 — 이동 얼굴이 목적지 편입 + 목적지 멤버 유지·id 승계 + 신규 앨범 없음",
    albums.get("b") == (0, 3, 4) and albums.get("a") == (1, 2) and not any(c.is_new for c in result.clusters),
  )

  # (n) 근중복 행 붕괴 (ADR-029) — event 8 재현: 동일 인물 5장(쌍 0.42~0.50, 병합 임계 0.55 미만
  # 대역) + 같은 사진 재업로드가 만든 완전 복제 5행. HDBSCAN은 밀도 기반이라 거리 0의 복제 쌍을
  # 실제 인물 덩어리보다 우선 선택해 앨범이 쌍 단위로 와해된다.
  dup_person = spread_vectors((0.72, 0.70, 0.68, 0.66, 0.64), axis(0), 10)
  dup_polluted = np.vstack([dup_person, dup_person.copy()])
  result = recluster(dup_polluted, [None] * 10, config=no_collapse)
  check(
    "(n) 전제: 붕괴 OFF면 재업로드 복제가 동일 인물을 여러 앨범으로 쪼갠다 (event 8 재현)",
    len(result.clusters) > 1,
  )
  result = recluster(dup_polluted, [None] * 10)
  check(
    "(n) 붕괴 ON(기본): 복제 오염 10행 → 앨범 1개 전원 소속, 복제 행은 대표의 LOO 공유",
    len(result.clusters) == 1
    and result.clusters[0].member_indices == tuple(range(10))
    and result.clusters[0].membership_similarities
    == tuple(result.clusters[0].membership_similarities[(i + 5) % 10] for i in range(10)),
  )
  # 직전 상태가 이미 쌍 앨범으로 쪼개져 있던 event: 붕괴 후 한 앨범이 최강 겹침 id를 승계하고,
  # 흡수된 나머지 쌍 앨범 id는 은퇴 목록으로 나가야 하류(Spring)가 유령 인물을 정리한다.
  dup_pair = spread_vectors((0.72, 0.70), axis(0), 10)  # 쌍 유사도 0.504
  shattered = np.vstack([dup_pair, dup_pair.copy()])
  result = recluster(shattered, ["pA", "pB", "pA", "pB"])
  check(
    "(n) 붕괴 후 최강 겹침 id 승계(동률은 등장 순) + 흡수된 쌍 앨범 id 은퇴",
    len(result.clusters) == 1
    and result.clusters[0].cluster_id == "pA"
    and not result.clusters[0].is_new
    and result.clusters[0].member_indices == (0, 1, 2, 3)
    and result.retired_cluster_ids == ("pB",),
  )
  # 사람 결정 우선 — 완전 복제 쌍이라도 사용자 cannot-link가 걸려 있으면 접지 않고 분리를 유지한다.
  twin = np.stack([axis(0), axis(0)]).astype(np.float32)
  result = recluster(twin, [None, None], Constraints(cannot_link=((0, 1),)))
  check(
    "(n) 복제 쌍 내부의 사용자 cannot-link는 붕괴를 막고 분리를 유지 (사람 결정 우선)",
    all(len(c.member_indices) == 1 for c in result.clusters),
  )
  # 제약 리매핑 — 복제 '행' 인덱스로 걸린 cannot-link가 대표로 이관되어 병합을 계속 차단한다:
  # A 복제쌍 + B 복제쌍(A↔B 0.72 — 제약 없으면 (k)처럼 한 앨범)에서 비대표 행끼리 (1, 3)을 지목.
  ab = np.stack([axis(0), axis(0), k_b, k_b]).astype(np.float32)
  result = recluster(ab, [None] * 4)
  check(
    "(n) 전제: 제약 없으면 복제쌍 붕괴 후 0.72 쌍은 한 앨범 + 전 행 복원",
    len(result.clusters) == 1 and result.clusters[0].member_indices == (0, 1, 2, 3),
  )
  result = recluster(ab, [None] * 4, Constraints(cannot_link=((1, 3),)))
  check(
    "(n) 비대표 복제 행끼리의 cannot-link도 대표로 이관돼 병합 차단 + 전 행 복원",
    {c.member_indices for c in result.clusters} == {(0, 1), (2, 3)},
  )
  # 고아 승격 경계 — 클러스터가 있는 이벤트에서 흡수에 실패한 복제 그룹은 승격하지 않고 노이즈로
  # 남긴다(쌍 앨범 부활 차단, 재업로드 = 같은 얼굴의 1회 등장). 인물 B 정상 3장 + 남남 A 복제쌍.
  b_rows = spread_vectors((0.9, 0.85, 0.8), axis(0), 10)
  a_far = (0.15 * axis(0) + math.sqrt(1.0 - 0.15**2) * axis(30)).astype(np.float32)
  mixed_dup = np.vstack([b_rows, a_far[None, :], a_far[None, :]])
  result = recluster(mixed_dup, [None] * 5)
  check(
    "(n) 클러스터가 있는 이벤트의 미흡수 복제 그룹은 쌍 앨범으로 승격하지 않고 uncertain",
    len(result.clusters) == 1 and result.clusters[0].member_indices == (0, 1, 2) and result.noise_indices == (3, 4),
  )

  # (g) 설정 불변식 — 잘못된 임계 조합은 생성 시점에 거부
  check(
    "(g) floor > promote 조합 거부",
    raises_value_error(lambda: ClusterConfig(blob_promote_floor=0.5, blob_promote_similarity=0.45)),
  )
  check(
    "(g) floor < min_membership_similarity 조합 거부 (승격 즉시 재강등 churn 차단)",
    raises_value_error(lambda: ClusterConfig(blob_promote_floor=0.3)),
  )
  check(
    "(g) margin 활성 시 ratio < 1 거부 (형제 모호 통과 오설정 차단)",
    raises_value_error(lambda: ClusterConfig(margin_rescue_floor=0.4, margin_rescue_ratio=0.9)),
  )

  # (o) margin 구제 (2026-07-23 실측 재현) — 인물 A·B 각 3장 + 하드 포즈 얼굴 M(A와만 0.43 근접,
  # B와 0 — 옆얼굴 실측 top1 0.44~0.49/top2 0.14~0.28의 합성 재현) + 형제 모호 얼굴 X(A 0.43·B 0.42
  # — 배율 ≈1.0). 둘 다 HDBSCAN이 A에 붙였다가 회색지대 게이트(최강 쌍 <0.45)가 강등하는 실경로를
  # 밟는다 — margin 게이트는 그 강등분을 재심해 여유가 큰 M만 편입해야 한다.
  margin_a = spread_vectors((0.9, 0.88, 0.86), axis(0), 10)
  margin_b = spread_vectors((0.9, 0.88, 0.86), axis(1), 20)
  margin_m = (0.45 * axis(0) + math.sqrt(1.0 - 0.45**2) * axis(40)).astype(np.float32)
  margin_x = (0.45 * axis(0) + 0.44 * axis(1) + math.sqrt(1.0 - 0.45**2 - 0.44**2) * axis(41)).astype(np.float32)
  margin_event = np.vstack([margin_a, margin_b, margin_m[None, :], margin_x[None, :]])
  result = recluster(margin_event, [None] * 8)
  check(
    "(o) 전제: margin OFF(기본)면 하드 포즈 M·모호 X 둘 다 노이즈",
    {6, 7} <= set(result.noise_indices) and len(result.clusters) == 2,
  )
  margin_on = ClusterConfig(margin_rescue_floor=0.40, margin_rescue_ratio=1.7)
  result = recluster(margin_event, [None] * 8, config=margin_on)
  check(
    "(o) margin ON: 여유 큰 M만 top1(A)에 편입, 배율 ≈1.0인 X는 노이즈 유지",
    any(c.member_indices == (0, 1, 2, 6) for c in result.clusters) and 7 in result.noise_indices,
  )
  result = recluster(np.vstack([margin_a, margin_m[None, :]]), [None] * 4, config=margin_on)
  check(
    "(o) 군집 1개 이벤트는 top2가 없어 margin 구제 전체 건너뜀 (무차별 편입 방지)",
    3 in result.noise_indices,
  )

  print(f"\n합성 자가 검증 {passed}건 전부 통과")
