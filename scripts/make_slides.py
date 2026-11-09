# 발표자료 생성 v2 — 연구배경 · 방법 · 결과 (python-pptx, 16:9). 최소 글자: 본문 15pt, 표 12pt, 캡션 12pt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
R="/SPECPHYS_ROOT/_workspace/data/results"; S="/SPECPHYS_ROOT/_workspace/slides"
OUT=f"{S}/VirtualFace_rPPG_발표.pptx"
INK=RGBColor(0x1f,0x23,0x28); MUTED=RGBColor(0x6b,0x66,0x63); RED=RGBColor(0xb3,0x31,0x2e); GREEN=RGBColor(0x2e,0x7d,0x4f); SURF=RGBColor(0xf1,0xef,0xec); WHITE=RGBColor(0xff,0xff,0xff)
FONT="Noto Sans CJK JP"
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); W=prs.slide_width; H=prs.slide_height; blank=prs.slide_layouts[6]
def tb(slide,x,y,w,h,text,size=18,bold=False,color=INK,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP):
    box=slide.shapes.add_textbox(x,y,w,h); tf=box.text_frame; tf.word_wrap=True; tf.vertical_anchor=anchor; tf.margin_left=tf.margin_right=Inches(0.05); tf.margin_top=tf.margin_bottom=Inches(0.03)
    for i,line in enumerate(text if isinstance(text,list) else [text]):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=align; r=p.add_run(); r.text=line; r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=color; r.font.name=FONT
    return box
def bullets(slide,x,y,w,h,items,size=15,color=INK,gap=6):
    box=slide.shapes.add_textbox(x,y,w,h); tf=box.text_frame; tf.word_wrap=True; tf.margin_left=Inches(0.05)
    for i,it in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.space_after=Pt(gap); txt,lvl=(it if isinstance(it,tuple) else (it,0)); p.level=lvl
        r=p.add_run(); r.text=("•  " if lvl==0 else "–  ")+txt; r.font.size=Pt(size-1*lvl); r.font.color.rgb=color; r.font.name=FONT
    return box
def rect(slide,x,y,w,h,fill=SURF):
    s=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,x,y,w,h); s.fill.solid(); s.fill.fore_color.rgb=fill; s.line.fill.background(); s.shadow.inherit=False; return s
def header(slide,title,kicker=None,n=None):
    rect(slide,0,0,W,Inches(0.08),RED)
    if kicker: tb(slide,Inches(0.6),Inches(0.22),Inches(10),Inches(0.35),kicker,13,color=RED,bold=True)
    tb(slide,Inches(0.6),Inches(0.48),Inches(12.2),Inches(0.8),title,26,bold=True)
    if n: tb(slide,Inches(12.2),Inches(7.05),Inches(0.9),Inches(0.35),str(n),11,color=MUTED,align=PP_ALIGN.RIGHT)
def pic(slide,path,x,y,w=None,h=None): return slide.shapes.add_picture(path,x,y,width=w,height=h)
def table(slide,x,y,w,rows,col_w=None,size=12,hl_rows=(),hl_cols=(),bold_cells=()):
    nr,nc=len(rows),len(rows[0]); shp=slide.shapes.add_table(nr,nc,x,y,w,Inches(0.34)*nr); t=shp.table
    if col_w:
        for j,cw in enumerate(col_w): t.columns[j].width=Inches(cw)
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            c=t.cell(i,j); c.text=""; p=c.text_frame.paragraphs[0]; r=p.add_run(); r.text=str(val); r.font.size=Pt(size); r.font.name=FONT
            hl=(i in hl_rows) or (j in hl_cols and i>0) or ((i,j) in bold_cells); r.font.bold=(i==0) or hl; r.font.color.rgb=WHITE if i==0 else (RED if hl else INK)
            c.fill.solid(); c.fill.fore_color.rgb=INK if i==0 else (SURF if i%2==0 else WHITE); c.margin_left=c.margin_right=Inches(0.05); c.margin_top=c.margin_bottom=Inches(0.025); p.alignment=PP_ALIGN.CENTER if j>0 else PP_ALIGN.LEFT
    return shp
def note(slide,text,y=Inches(6.85)): tb(slide,Inches(0.6),y,Inches(12.1),Inches(0.5),text,12,color=MUTED)
n=0
# 1 표지
s=prs.slides.add_slide(blank); n+=1; rect(s,0,0,W,H,INK); rect(s,Inches(0.6),Inches(2.2),Inches(0.12),Inches(2.4),RED)
tb(s,Inches(0.9),Inches(2.1),Inches(11.5),Inches(1.6),"가상얼굴에 물리 기반 맥파를 주입해\n학습 분포의 결함을 메우는 rPPG",34,bold=True,color=WHITE)
tb(s,Inches(0.9),Inches(3.8),Inches(11.5),Inches(0.9),"Filling the Training Distribution: Physics-Injected Virtual Faces Repair Cross-Dataset Failure of Learned rPPG",16,color=RGBColor(0xc9,0xc5,0xc0))
tb(s,Inches(0.9),Inches(5.3),Inches(11),Inches(0.9),["연구 결과 발표  ·  2026-08-28","6개 실데이터셋(UBFC · PURE · MMPD · MCD-rPPG · DLCN · MR-NIRP) · PhysFormer / RhythmFormer / FactorizePhys"],14,color=RGBColor(0xc9,0xc5,0xc0))
# 2 한 장 요약
s=prs.slides.add_slide(blank); n+=1; header(s,"한 장 요약","TL;DR",n)
for i,(t,d,c) in enumerate([("문제","UBFC 한 데이터셋으로 학습된 rPPG 딥러닝 모델은 다른 데이터셋에서 무너진다\n(PhysFormer: MCD 29.3 BPM, PURE 16.5 BPM — 4개 아키텍처가 공통으로)",RED),("진단","붕괴는 이름 붙일 수 있는 세 실패 모드(harmonic doubling · flicker lock-in · BPM coverage gap)이고,\n원인은 아키텍처가 아니라 학습 데이터 분포의 결함",INK),("방법","가상얼굴 + 픽셀 단위 물리 pulse injection으로 파형 harmonics · 심박 대역 · 교란(nuisance)을\n제어 변수로 채운 합성 코퍼스 → 소규모 실데이터와 함께 학습(co-training)",INK),("결과","같은 PhysFormer가 doubling/halving 0건, MCD 29.3→0.48, PURE 16.5→0.57, MMPD에서 고전 POS 상회.\n추가 발견: 공개 트레이너의 HR 라벨 자체가 2배 심박(2f)을 가르치고 있었다(MMPD 52%)",GREEN)]):
    y=Inches(1.45+i*1.38); rect(s,Inches(0.6),y,Inches(1.6),Inches(1.2),c); tb(s,Inches(0.6),y,Inches(1.6),Inches(1.2),t,20,bold=True,color=WHITE,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
    rect(s,Inches(2.2),y,Inches(10.5),Inches(1.2),SURF); tb(s,Inches(2.35),y,Inches(10.3),Inches(1.2),d,15,anchor=MSO_ANCHOR.MIDDLE)
# 3 용어 설명
s=prs.slides.add_slide(blank); n+=1; header(s,"이 발표에서 쓰는 약어·용어","용어 설명 (표를 읽기 전에)",n)
table(s,Inches(0.6),Inches(1.4),Inches(6.0),[["모델 이름","뜻"],["base / PhysFormer(UBFC)","공개 가중치: UBFC 42명으로만 학습된 PhysFormer"],["T2L","합성 데이터만으로 학습(실데이터 없음) → 실데이터 zero-shot"],["CoTN","UBFC 1~41 + 합성 + 노이즈 증강 co-training"],["CoTNH","CoTN + 파형 harmonic 다양화"],["CoTNHM","CoTNH + 고심박 합성 + MCD 학습 정체성"],["CoTA","CoTNHM + PURE 학습 분할(01~06)"],["CoTAF (제안 최종)","CoTA + 저심박 합성 — 203클립, zero-shot 주 모델"],["CoTAF2","CoTAF + 운전 스트레스 합성(209클립)"],["CoTAM / CoTAM-L","CoTAF 풀 + MMPD 학습 분할(1~6) / -L = 라벨 수정판"],["-L 접미사","트레이너 HR 라벨을 수정(cumsum)해 재학습한 모델"],["RFA / FFA","RhythmFormer / FactorizePhys를 같은 코퍼스로 fine-tuning"],["POS / CHROM","학습이 없는 고전 신호처리 방법(기준선)"]],col_w=[1.9,4.1],size=12)
table(s,Inches(6.9),Inches(1.4),Inches(5.8),[["평가 용어","뜻"],["zero-shot","그 데이터셋의 어떤 영상도 학습에 쓰지 않음"],["held-out","같은 데이터셋이지만 학습에 없는 피험자(예: PURE 07~10)"],["in-domain","학습에 그 데이터셋이 포함됨(피험자/정체성은 분리)"],["MCD 안정 / 운동후","MCD-rPPG 정면 20명, 안정 상태 / 운동 직후 촬영"],["MR-NIRP 정지·소·대동작","실차 운전 중 촬영, 머리 움직임 크기 3단계"],["UBFC + flicker / colour-temp","실데이터 held-out에 통제된 조명 깜빡임 / 색온도 변화"],["합성 C3 / E3","합성 평가셋(학습 범위 안 노이즈 / 범위 밖 외삽)"],["MAE / RMSE / MAPE / r","심박 오차(BPM) 평균 / 제곱평균 / 백분율 / 상관계수"],["'0.00'의 뜻","정본 FFT는 0.88 BPM 칸 단위 → 전 영상이 정답 칸과 일치(1 BPM 이내). 고해상 값은 별도 표기"],["doubling / halving","예측이 정답의 2배 / 절반으로 잡힌 영상 수(실패 모드)"],["harmonic doubling","맥파의 2차 배음이 커져 2배 심박을 출력하는 실패"],["flicker lock-in","주변광의 깜빡임 주기를 심박으로 읽는 실패"],["BPM coverage gap","학습 데이터에 없던 심박 대역에서 생기는 오류"]],col_w=[2.1,3.7],size=12)
# 4 배경
s=prs.slides.add_slide(blank); n+=1; header(s,"rPPG: 카메라만으로 심박을 읽는다 — 그러나 데이터셋을 바꾸면 무너진다","연구 배경",n)
bullets(s,Inches(0.6),Inches(1.45),Inches(6.3),Inches(5.5),["원격 광용적맥파(rPPG): 얼굴 피부색의 약 1% 미세 변동에서 심박(HR)을 복원","고전 방법(POS/CHROM) → 딥러닝(PhysNet, PhysFormer, RhythmFormer, FactorizePhys …)","같은 데이터셋 안에서는 성능 포화(UBFC/PURE 오차 1 BPM 미만)","그러나 다른 데이터셋으로 가면:",("UBFC 학습 PhysFormer → MCD-rPPG 29.3 BPM (원저자도 43.7 관측)",1),("→ PURE held-out 16.5 BPM, 실차 운전 13~17 BPM",1),("고전 POS가 오히려 안정적(4.75 / 2.74)",1),"공개 데이터셋은 작다(UBFC 42명, PURE 10명) → 파형·심박·조명 분포가 좁음","질문: 이 붕괴는 모델의 한계인가, 데이터의 한계인가?"],15)
tb(s,Inches(7.2),Inches(1.45),Inches(5.5),Inches(0.4),"'UBFC만' 학습한 공식 가중치의 cross-dataset HR MAE (BPM)",13,bold=True,color=MUTED)
table(s,Inches(7.2),Inches(1.85),Inches(5.5),[["모델 (UBFC만)","MCD 안정","PURE","운전 정지","MMPD"],["PhysFormer","29.27","16.50","13.41","14.39"],["RhythmFormer","17.45","7.47","11.24","15.26"],["EfficientPhys","11.78","8.37","10.78","–"],["PhysNet","43.46","9.94","15.27","–"],["FactorizePhys","1.36","4.60","5.37","13.47"],["POS (학습 없음)","4.75","2.74","15.08","12.42"]],col_w=[1.9,0.9,0.85,0.95,0.9],size=12)
tb(s,Inches(7.2),Inches(4.6),Inches(5.5),Inches(1.6),"→ 대부분의 아키텍처가 같은 축(MCD)에서 함께 무너진다.\n    FactorizePhys만 예외 → 아키텍처가 아닌 공통 요인이 있다",14,color=RED)
# 5 실패 모드
s=prs.slides.add_slide(blank); n+=1; header(s,"붕괴에는 이름이 있다 — 세 가지 실패 모드","연구 배경 · 진단",n)
pic(s,f"{R}/fig1_v4_failures.png",Inches(0.6),Inches(1.4),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.5),Inches(12.1),Inches(1.8),["harmonic doubling: 맥파의 2차 배음이 기본파보다 커져 2배 심박을 출력 — MCD 20영상 중 8건","flicker lock-in: 주변광 깜빡임 주기(1.1 Hz 등)를 심박으로 읽음 — 실데이터+통제 flicker에서 30.3 BPM","BPM coverage gap: 학습 분포 밖 심박 → 저심박은 doubling, 고심박은 halving(PURE·운동 직후)","→ 세 모드 모두 '학습 데이터에 없던 것'으로 설명된다 = 학습 분포의 결함"],15)
# 6 접근
s=prs.slides.add_slide(blank); n+=1; header(s,"접근: 부족한 분포를 가상얼굴로 설계해서 채운다","방법 · 개요",n)
pic(s,f"{R}/fig1_overview.png",Inches(0.6),Inches(1.4),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.4),Inches(12.1),Inches(1.9),["존재하지 않는 정체성(SFHQ / SDXL-Turbo) → LivePortrait 애니메이션 → 30fps 무손실 영상","피부 화소에 물리 기반 맥파 주입  I ← I·(1 + α·A(x,y)·p(t)·P_bv)  — 정답(ground truth)이 구성상 정확","제어 변수: 파형 harmonics, 심박 대역(42~152 BPM), 교란(flicker·drift·색온도·강체 운동·센서·mpeg4·관류 이질성)","합성 코퍼스 + 소규모 실데이터를 함께 학습 → 동일 백본, 동일 설정(10 epoch, LR 5e-5)"],15)
# 7 주입 예시
s=prs.slides.add_slide(blank); n+=1; header(s,"픽셀 단위 pulse injection: 정체성 · 애니메이션 · 주입 · 조명 변화","방법 · 데이터 생성",n)
pic(s,f"{R}/fig_identities_anim_ppg_illum.png",Inches(1.4),Inches(1.35),w=Inches(10.5)); note(s,"각 행 = 비실존 정체성 1명. 변조 약 1%(×40 확대 표시). 조명은 주입 후 적용. 실제 달성된 BPM을 측정해 라벨로 사용.",Inches(6.75))
# 8 nuisance 데모 영상
s=prs.slides.add_slide(blank); n+=1; header(s,"교란(nuisance) 데모: 같은 정체성·같은 주입에 12종 교란을 각각 적용 (10초, 클릭 재생)","방법 · 노이즈 증강",n)
s.shapes.add_movie(f"{S}/nuisance_demo.mp4",Inches(0.6),Inches(1.35),width=Inches(6.1),height=Inches(5.68),poster_frame_image=f"{S}/nuisance_demo_frame.png",mime_type="video/mp4")
bullets(s,Inches(7.0),Inches(1.4),Inches(5.8),Inches(5.6),["① clean(주입 ×1, 눈에 안 보임) ② 주입 ×40 과장 ③ 관류 이질성 A(x,y)","④ illumination flicker 1.1 Hz(심박 대역 → flicker lock-in의 원인) ⑤ illumination drift ⑥ colour-temperature drift","⑦ 강체 운동 random walk ⑧ periodic nod ⑨ scale/vibration jitter","⑩ sensor noise ⑪ mpeg4 compression(q6) ⑫ driving stress = 전부 복합","적용 순서는 코퍼스 빌더와 동일: 주입 → 운동 → 조명 → 색온도 → 센서. 맥동은 픽셀에 있어 얼굴과 함께 움직인다","MMPD 조건과의 대응: 조명 4종 ↔ ④⑤⑥, Rotation/Talking ↔ ⑦⑧. Walking(카메라 자체 흔들림)은 미포함 → 한계"],14)
# 9 코퍼스 설계
s=prs.slides.add_slide(blank); n+=1; header(s,"코퍼스 설계: 실패 모드 하나당 채워 넣은 축 하나","방법 · 코퍼스",n)
table(s,Inches(0.6),Inches(1.45),Inches(12.1),[["실패 모드","채운 축 (합성 코퍼스에 추가한 것)","관측된 효과: 축 추가 전 → 후 HR MAE (BPM), 평가 셋"],["flicker lock-in","조명 flicker/drift + 색온도 (합성·실데이터 양쪽 노이즈 증강)","30.32 → 0.00 [고해상 29.5 → 2.6]  (UBFC held-out + 1.1 Hz flicker 8영상, base → CoTN)"],["harmonic doubling","해석적 harmonic 파형 p(θ)=sinθ+a₂sin(2θ+φ₂)+a₃sin(3θ+φ₃), a₂ 균등 스윕","42.63 → 5.19, 2배 출력 8 → 0건  (MCD 안정 20영상, CoTN → CoTNH)"],["high-HR halving","122~152 BPM 합성 보강 (+ MCD 학습 정체성)","19.9 → 5.5  (MCD 운동 직후 20영상, CoTNH → CoTNHM)"],["low-HR doubling","42~50 BPM 합성 보강 + PURE 학습 분할 01~06","2배 출력 14 → 0건  (PURE 59영상, CoTNHM → CoTA/CoTAF)"],["압축 아티팩트","mpeg4 재인코딩 사본","2×2 ablation: 파형 p<.001 · 압축 p=.022 · 상호작용 없음 = 두 축의 효과가 독립적으로 더해짐"]],col_w=[2.0,5.0,5.1],size=12)
bullets(s,Inches(0.6),Inches(4.4),Inches(12.1),Inches(2.6),["최종 학습 풀(CoTAF, 203클립): UBFC 1~41 + MCD 정체성 21~44 + PURE 01~06 + 합성(harmonic·고심박·저심박·노이즈·압축)","분할 규칙: MCD 평가 20명은 학습 정체성과 분리 · PURE 07~10, UBFC 42~49, MMPD 7~33은 held-out · DLCN, MR-NIRP는 순수 zero-shot","평가: 영상당 심박 1개(rPPG-Toolbox 정본 프로토콜) · doubling |비율−2|<0.2, halving |비율−0.5|<0.1"],14)
# 10 데이터셋
s=prs.slides.add_slide(blank); n+=1; header(s,"평가 데이터셋 6종 — 조명 · 동작 · 심박 · 피부톤 · 촬영 환경을 모두 덮는다","실험 설정",n)
table(s,Inches(0.6),Inches(1.45),Inches(12.1),[["데이터셋","피험자","평가 클립","조건","역할"],["UBFC-rPPG","42","8 (42~49)","실내 정면","사전학습 1~41 / held-out"],["PURE","10","24 (07~10)","6가지 머리 동작","학습 01~06 / held-out"],["MMPD (신규)","33","540 (7~33)","조명 4 × 동작 4 × 운동 × 피부톤, 모바일 320×240","학습 1~6 / held-out"],["MCD-rPPG","600","20 + 20","안정 / 운동 직후","정체성 21~44 학습 / 20명 평가"],["DLCN","98","20","야간 동적 조명","순수 zero-shot"],["MR-NIRP Car","19","19 × 3","실차 운전 정지 / 소동작 / 대동작","순수 zero-shot"]],col_w=[1.9,1.0,1.5,4.6,3.1],size=13)
bullets(s,Inches(0.6),Inches(4.7),Inches(12.1),Inches(2.3),["비교 대상: PhysFormer(주 백본) · RhythmFormer · FactorizePhys · EfficientPhys · PhysNet(공식 UBFC 가중치) · POS/CHROM(고전)","지표: MAE / RMSE / MAPE / Pearson r + doubling · halving 건수(실패 모드를 표에 노출)","통계: 영상 단위 대응 Wilcoxon 검정, 코퍼스 단조성 Friedman + Page L(순수 zero-shot 77영상: p=8.7×10⁻⁶)"],14)
# 11 종합 결과표
s=prs.slides.add_slide(blank); n+=1; header(s,"종합 결과: 9개 평가 축 × 주요 모델 HR MAE (BPM, 낮을수록 좋음)","결과 1 · 종합 표",n)
rows=[["평가 축","PhysFormer\nbase (UBFC)","CoTAF\n(제안, zero-shot)","CoTAM-L / CoTAN-CV-L\n(in-domain)","RhythmFormer\n(UBFC)","FactorizePhys\n(UBFC)","Efficient\nPhys","PhysNet","POS","CHROM"],
["PURE held-out 07~10","16.50","0.57","0.79","7.47","4.60","8.37","9.94","2.74","–"],["MCD 안정","29.27","0.48","1.36","17.45","1.36","11.78","43.46","4.75","–"],["MCD 운동 후","32.52","4.17","4.09","24.92","8.31","16.92","31.60","12.17","–"],["DLCN (zero-shot)","3.69","1.36","1.63","1.93","1.76","1.49","2.64","2.55","–"],["MR-NIRP 운전 정지 (held-out 7~19)","13.32","5.95","5.69 (CV)","15.48","7.17","–","–","11.56","–"],["MR-NIRP 운전 소동작 (held-out 7~19)","15.75","9.06","6.15 (CV)","16.70","11.36","–","–","13.45","–"],["MR-NIRP 운전 대동작 (held-out 7~19)","14.13","8.65","9.07 (CV)","16.02","12.30","–","–","7.64","–"],["MMPD held-out 7~33 (n=540)","14.39","11.56","9.98","15.26","13.47","–","–","12.42","13.06"],["UBFC held-out + flicker","30.32","0.00 (고해상 2.61)","3.63 (4.09)","29.00","28.89","–","–","–","–"]]
best={(1,2),(2,2),(3,3),(4,2),(5,3),(6,2),(7,8),(8,3),(9,2)}
table(s,Inches(0.6),Inches(1.4),Inches(12.1),rows,col_w=[2.5,1.15,1.2,1.25,1.15,1.15,0.9,0.85,0.75,0.8],size=12,bold_cells=best)
note(s,"빨간 굵은 글씨 = 해당 축 최저 오차. base·RhythmFormer·FactorizePhys·EfficientPhys·PhysNet = UBFC로만 학습한 공식 가중치(MCD·DLCN·MR-NIRP·MMPD에는 zero-shot). CoTAF는 MCD·PURE·UBFC를 피험자 분리로 학습(in-domain), DLCN·MR-NIRP·MMPD는 zero-shot. 정본 프로토콜은 64초 FFT의 0.88 BPM 칸 단위로 양자화되므로 '0.00' = 전 영상이 정답과 같은 칸(1 BPM 이내); 괄호는 8× zero-pad 고해상 FFT 값. CoTAM-L은 MMPD 1~6을 추가 학습; MR-NIRP 행의 CoTAM-L 열 자리에는 MR-NIRP in-domain 3-fold CV 모델(CoTAN-CV-L, 19명 전원 held-out) 값을 표기. 정본 프로토콜, 영상당 심박 1개.",Inches(6.55))
# 12 cross chart
s=prs.slides.add_slide(blank); n+=1; header(s,"같은 백본, 학습 분포만 바꿔 6개 데이터셋에서 고전·학습 기준선 상회","결과 1b · Cross-dataset",n)
pic(s,f"{S}/figA_cross.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(6.0),Inches(12.1),Inches(1.4),["CoTAF: MCD 29.27→0.48 / 32.52→4.17, PURE 16.50→0.57, DLCN 3.69→1.36, 운전 정지 13.41→7.49(POS 15.08, p=.016), MMPD 14.39→11.56(POS 12.42, p=.045)","유일한 열세: 운전 대동작 10.78 vs POS 9.48(통계적으로 비유의, p=.69) — 라벨 수정판 CoTAF-L은 9.44"],14)
# 13 계보
s=prs.slides.add_slide(blank); n+=1; header(s,"축을 하나씩 채울수록 단조 개선 — 어느 축이 어느 실패를 고쳤는가","결과 2 · 귀인",n)
pic(s,f"{S}/figD_lineage.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.8),Inches(12.1),Inches(1.6),["harmonic 다양화가 MCD 붕괴를 해소(42.6→5.2), 고심박·MCD 정체성 추가가 운동후를(19.9→5.5), PURE 분할+저심박이 PURE doubling을(14→0) 해소","2×2 ablation: 파형 주효과 p<.001, 압축 p=.022, 상호작용 없음(가산적). 실분포 정합 파형은 균등 스윕보다 나쁨","순수 zero-shot 77영상 단조 개선: Friedman p=8.7×10⁻⁶, Page z=4.48"],14)
# 14 MMPD
s=prs.slides.add_slide(blank); n+=1; header(s,"MMPD: 조명 4종 × 동작 4종 × 피부톤 — 교란 분류의 물리적 검증","결과 3 · MMPD (신규)",n)
pic(s,f"{S}/figB_mmpd_heat.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.3),Inches(12.1),Inches(2.1),["held-out 540클립: CoTAF 11.56 < POS 12.42 < FactorizePhys 13.47 < base 14.39 < RhythmFormer 15.26 — 조명 3/4·정지·운동후·피부톤 전 구간에서 POS 상회(zero-shot)","MMPD 학습 분할(1~6)을 넣고 라벨을 수정한 CoTAM-L: 9.98(POS 대비 p=1.7×10⁻⁷, doubling 0), 전 조건에서 POS 상회","RhythmFormer/FactorizePhys는 운동 후·Walking 고심박에서 halving 53~80건 — 심박 대역 보강 효과가 MMPD에서도 재확인","한계: Walking(보행 촬영)은 전 모델 19.7~31 — 카메라 자체 흔들림은 현재 교란 분류 밖"],13)
# 15 MMPD 공정
s=prs.slides.add_slide(blank); n+=1; header(s,"공정 비교: 백본별 'UBFC만' vs '동일 코퍼스로 fine-tuning' — MMPD 조건별","결과 3b · 백본 공정 비교",n)
pic(s,f"{S}/figE_mmpd_fair.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.75),Inches(12.1),Inches(1.7),["세 백본 모두 같은 풀(ALLF2 209클립)·같은 10 epoch — MMPD는 어느 모델도 학습에 쓰지 않음(zero-shot)","PhysFormer 14.39 → 11.75 개선 · RhythmFormer 15.26 → 16.91 악화(halving 60→80) · FactorizePhys 13.47 → 13.19 거의 불변","→ 코퍼스 효과는 백본 의존적(적용 가능하나 불변은 아님). 주의: RFA/FFA/CoTAF2는 batch 2, 백본별 하이퍼파라미터 미조정"],13)
# 16 UBFC 공정
s=prs.slides.add_slide(blank); n+=1; header(s,"UBFC-rPPG에서의 공정 비교: zero-shot · in-domain · held-out을 같은 표에","결과 3c · UBFC 조건별",n)
pic(s,f"{S}/figF_ubfc_fair.png",Inches(0.9),Inches(1.35),w=Inches(11.5))
bullets(s,Inches(0.6),Inches(6.15),Inches(12.1),Inches(1.3),["열: UBFC 42명(clean) · held-out 42~49(clean) · +flicker · +colour-temp. 행: 백본별 PURE만(zero-shot) / UBFC만(in-domain) / +동일 코퍼스(held-out)","clean은 모두 쉬움(0.4~2.3); 차이는 통제 교란에서만 — 공식 가중치 3백본 모두 flicker lock-in(19~30), 코퍼스로 PhysFormer·RhythmFormer 해소, FactorizePhys는 불변"],13)
# 16b MR-NIRP in-domain
s=prs.slides.add_slide(blank); n+=1; header(s,"MR-NIRP(실차 운전)에서 같은 실험: CoTAF zero-shot vs CoTAN-L(피험자 1~6 in-domain + 라벨 수정)","결과 3d · MR-NIRP 조건별",n)
pic(s,f"{S}/figG_nirp.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.75),Inches(12.1),Inches(1.7),["CoTAN-L(1~6 학습, held-out 7~19): 5.48 / 10.95 / 9.40 — zero-shot CoTAF(5.95 / 9.06 / 8.65)와 구분 안 됨(p=.54~.91). 18클립 in-domain은 새 피험자로 이전되지 않음","학습 피험자를 12~13명으로 늘린 3-fold CV(CoTAN-CV-L, 19명 전원 held-out): 5.69 / 6.15 / 9.07 — 정지·소동작은 POS 대비 개선(p=.003 / .09), CoTAF 대비는 비유의; 문헌 in-domain PhysFormer(Bouraffa 5-fold 6.35 / 7.88 / 9.17)와 동급","대동작은 학습 피험자를 2배로 늘려도 9.07(POS 9.48, 비유의) — fold별 2.8~13.2로 피험자 편차가 지배. 카메라 흔들림 합성(EGO6, CoTAF3-L)도 실패(13.46, MMPD 악화 p<.001) → 실차 대동작의 지배 요인(머리 out-of-plane 회전·상자 이탈)은 아직 코퍼스 밖"],13)
# 17 라벨 발견
s=prs.slides.add_slide(blank); n+=1; header(s,"추가 발견: 학습 라벨 자체가 harmonic doubling을 가르치고 있었다","결과 4 · 라벨 harmonic ambiguity",n)
pic(s,f"{S}/figC_label.png",Inches(0.6),Inches(1.35),w=Inches(12.1))
bullets(s,Inches(0.6),Inches(5.6),Inches(12.1),Inches(1.8),["공개 트레이너는 미분된 라벨 청크에 Welch 스펙트럼을 적용해 심박 라벨을 뽑음 → 2차 배음이 부풀어 라벨이 2배 심박(2f)이 됨: MMPD 51.7%, PURE 19%, MCD 20%, UBFC 10%, 합성 0%","적분 → detrend → 대역통과 후 추출(RPPG_HR_LABEL=cumsum)하면 0%. 같은 데이터로 CoTAM: 학습 피험자 22.31(doubling 27) → 1.78, held-out 13.95 → 9.98","해석: 합성 코퍼스(라벨 오염 0%)가 효과를 낸 메커니즘의 일부 = 실측 라벨 오염의 희석"],13)
# 18 라벨 수정 재학습
s=prs.slides.add_slide(blank); n+=1; header(s,"라벨을 고치고 전 계보를 다시 학습하면? — 효과는 '오염이 지배적인 곳'에 국한","결과 5 · 라벨 수정 재학습 (10모델)",n)
table(s,Inches(0.6),Inches(1.45),Inches(12.1),[["모델 (원래 → 라벨 수정 -L)","MCD 안정","MCD 운동후","DLCN","운전 대동작","합성 노이즈 C3","MMPD held-out"],["CoTN (harmonic 다양화 전)","42.63 → 18.41 (p=.009)","32.65 → 22.32","– → 6.15","– → 11.5","0.06 → 0.06","– → 14.13"],["CoTNH (harmonic 다양화)","5.19 → 8.44","19.91 → 16.17","6.68 → 6.72","10.3 → 8.1","0.18 → 0.18","14.78 → 13.19"],["CoTAF (최종 zero-shot)","0.48 → 0.57","4.17 → 4.61","1.36 → 2.42","10.8 → 9.4","0.06 → 0.06","11.56 → 12.19"],["CoTAM (+MMPD 1~6)","1.14 → 1.36","4.31 → 4.09","2.68 → 1.63","19.4 → 7.8","33.46 → 0.12","13.95 → 9.98"],["CoTB (균형 풀)","2.15 → 1.05","3.65 → 4.92","– → 2.64","– → 9.7","17.58 → 0.12","– → 13.52"],["CoTBH (균형 + 주파수 벌점)","1.10 → 1.89","5.41 → 4.83","– → 1.54","– → 10.2","17.52 → 0.12","– → 12.91"]],col_w=[2.7,2.1,1.6,1.4,1.4,1.5,1.4],size=12,hl_rows=(1,4,5,6))
bullets(s,Inches(0.6),Inches(4.75),Inches(12.1),Inches(2.3),["결정적 개선 3곳: CoTN의 MCD doubling 붕괴(라벨 원인), CoTAM의 합성·MMPD 붕괴, 균형 풀의 합성 노이즈 붕괴(17.5 → 0.1)","→ 이전에 'clean 희석 부작용'으로 해석했던 균형 풀 현상은 라벨 문제였음(해석 철회). harmonic 코퍼스가 이미 들어간 계보는 축별 ±, 유의차 없음","PURE의 low-HR doubling은 라벨 수정 후에도 그대로 → 심박 대역 커버리지 원인 확정. 주 모델은 CoTAF 유지, 이후 학습은 모두 수정 라벨"],13)
# 19 백본 전이
s=prs.slides.add_slide(blank); n+=1; header(s,"같은 코퍼스를 다른 백본에: 적용 가능하지만 불변은 아니다","결과 6 · 백본 전이",n)
table(s,Inches(0.6),Inches(1.45),Inches(7.4),[["백본 (UBFC만 → +코퍼스)","MCD 안정","MCD 운동후","DLCN","운전 대동작","MMPD"],["PhysFormer","29.27 → 0.48","32.52 → 4.17","3.69 → 1.36","16.75 → 10.78","14.39 → 11.56"],["RhythmFormer","17.45 → 3.87","24.92 → 4.13","1.93 → 13.27","16.14 → 18.32","15.26 → 16.91"],["FactorizePhys","1.36 → 1.58","8.31 → 7.38","1.76 → 1.76","10.96 → 11.06","13.47 → 13.19"]],col_w=[2.0,1.1,1.15,1.05,1.1,1.0],size=12)
bullets(s,Inches(8.3),Inches(1.45),Inches(4.5),Inches(5.5),["PhysFormer: 전 축 변혁적 개선","RhythmFormer: 붕괴 축(MCD, flicker 29.0→5.0)은 해소되나 미학습 축(DLCN·운전) 악화 — 백본별 설정 미조정","FactorizePhys: 거의 불변 — 사전학습부터 강건하고 코퍼스에도 반응하지 않음","RhythmFormer의 라벨 경로는 정식 후처리라 라벨 결함 무관 → 재학습 불필요","결론: 코퍼스는 '적용 가능'하나 '백본 불변'은 아님 — 정직하게 병기"],14)
tb(s,Inches(0.6),Inches(3.4),Inches(7.4),Inches(3.3),"PhysFormer++는 공개 코드·가중치가 없어 측정 불가(원 논문도 PURE/UBFC 수치 없음).\n\n문헌 앵커: 우리 POS 측정치가 원저자 보고(MCD 3.80, Bouraffa 운전 대동작 9.16, DLCN 교차 3.94)와 정합 → 파이프라인 건전성 확인. 평가 프로토콜 계열(toolbox vs SiNC 윈도우)은 섞지 않음.",13,color=MUTED)
# 20 결론
s=prs.slides.add_slide(blank); n+=1; header(s,"결론과 기여","정리",n)
for i,(t,d) in enumerate([("기여 1  실패 모드의 발견·명명·수리","harmonic doubling · flicker lock-in · BPM coverage gap(양방향) · domain shortcut — 각각을 제어된 가상얼굴 축으로 재현하고 메웠다"),("기여 2  물리 기반 가상얼굴 코퍼스","존재하지 않는 정체성 + 픽셀 단위 pulse injection + 교란 분류: 정답이 구성상 정확하고 분포를 설계할 수 있다"),("기여 3  6개 실데이터셋 검증","동일 PhysFormer가 doubling/halving 0, 13개 축 중 12개에서 고전 상회; MMPD 조명·동작·피부톤 조건별로 POS 상회"),("기여 4  학습 라벨의 harmonic ambiguity","공개 트레이너의 심박 라벨이 2f를 가르치고 있었다(MMPD 52%). 수정 패치 + 전 계보 재학습으로 효과 범위를 정직하게 확정")]):
    y=Inches(1.45+i*1.18); rect(s,Inches(0.6),y,Inches(12.1),Inches(1.05),SURF); rect(s,Inches(0.6),y,Inches(0.1),Inches(1.05),RED)
    tb(s,Inches(0.85),y+Inches(0.05),Inches(11.7),Inches(0.4),t,16,bold=True); tb(s,Inches(0.85),y+Inches(0.45),Inches(11.7),Inches(0.6),d,13,color=MUTED)
tb(s,Inches(0.6),Inches(6.25),Inches(12.1),Inches(1.0),"한계·다음 단계: 운전 대동작·MMPD Walking(카메라 흔들림은 교란 분류 밖) · 라벨 없는 도메인 적응은 실 PPG 파형 뱅크(TW)로 일부 달성(−2.0 BPM); Skin 6·Walking은 실 라벨 상한도 못 풀어 남음(움직임-조명 결합·모션 블러 미구현) · 백본별 하이퍼파라미터 · 단일 시드 ablation · 전 계보 수정 라벨 기준으로 논문 v2 통일",13,color=RED)
# 합성만 학습
s=prs.slides.add_slide(blank); n+=1; header(s,"합성 데이터만으로 학습하면 zero-shot이 되는가? — 부분적으로만","결과 7 · 합성만 학습 (SYNALL)",n)
table(s,Inches(0.6),Inches(1.45),Inches(12.1),[["모델 (학습 데이터)","UBFC 42–49","+flicker","PURE 07–10","MCD 안정","MCD 운동후","DLCN","운전 정지/소/대","MMPD 7–33"],["합성만 T2L (20클립, UBFC 초기화)","1.87","18.35","20.87 (dbl 10)","46.89 (dbl 6)","50.14","–","–","–"],["합성만 SYNALL-FT (68클립, UBFC 초기화)","1.54","11.87","4.41","31.29 (dbl 6)","37.92","5.84","8.51 / 8.23 / 11.15","16.52 (hlv 46)"],["합성만 SYNALL-SCR (68클립, 무작위 초기화 30ep)","36.25","28.12","21.51","23.77","33.88","17.80","11.98 / 18.73 / 17.53","23.32 (hlv 120)"],["PhysFormer — UBFC만","1.43","30.32","16.50 (dbl 8)","29.27","32.52","3.69","13.41 / 16.93 / 16.75","14.39"],["CoTAF (실 정체성 161 + 가상얼굴 42)","1.10","0.00","0.57","0.48","4.17","1.36","7.49 / 9.11 / 10.78","11.56"],["POS (학습 없음)","–","–","2.74","4.75","12.17","2.55","15.08 / 13.04 / 9.48","12.42"]],col_w=[3.0,1.0,0.9,1.2,1.2,1.1,0.9,1.6,1.2],size=12,bold_cells={(2,3),(2,7)})
bullets(s,Inches(0.6),Inches(4.6),Inches(12.1),Inches(2.5),["합성이 대체하는 것(커버리지): 심박 대역·저심박(PURE 16.5 → 4.4, doubling 0), 정의된 교란(운전 정지/소동작 8.5/8.2로 POS 상회, flicker 30 → 12)","합성이 대체하지 못한 것: MCD harmonic doubling(31.3 — 실 UBFC와의 co-training이 있어야 5.2), DLCN 야간 외형(5.8 > 3.7), MMPD 모바일 외형·피부톤(16.5, halving 46)","무작위 초기화 합성만(68클립)은 표현 학습 자체가 안 됨(UBFC 36) → '엄밀한 합성만'은 이 규모에서 불성립. 다음 단계: 각 데이터셋 정체성의 정지 프레임을 애니메이션·주입해 외형 도메인을 합성으로 흡수"],13)
# 도메인 특화 합성 설계
s=prs.slides.add_slide(blank); n+=1; header(s,"도메인 특화 합성: 대상 데이터셋의 심박·노이즈 특성을 재고, 그 범위를 '포함'하도록 합성한다","방법 강화 · 도메인 적응 (MMPD 사례)",n)
table(s,Inches(0.6),Inches(1.45),Inches(12.1),[["특성","측정 방법 (라벨 필요?)","MMPD 학습 분할 측정 p10 / p50 / p90","기존 합성 범위","공백"],["심박","학습 분할 GT, 정본 프로토콜 (O)","65 / 78 / 102 BPM","42~152","없음"],["2차 harmonic 비 a₂","GT PPG 스펙트럼 2f/f (O)","0.34 / 0.57 / 1.10","0~0.75","p50 이상 밖"],["3차 harmonic 비 a₃","3f/f (O)","0.23 / 0.36 / 0.55","0~0.28","대부분 밖"],["조명 주기 성분","화면 가장자리 밝기 스펙트럼 (X)","0.53 Hz, 진폭 ≤0.03","1.0~1.3 Hz","심박 대역 flicker 거의 없음"],["조명 드리프트","동일 (X)","0.03 / 0.28 / 1.17","0.1~0.22","p50부터 밖 (Walking·Nature)"],["얼굴 이동 (얼굴폭 256 기준)","Haar 상자 궤적 (X)","5 / 10 / 86 px","6~20 px","Walking 구간 밖"],["센서 노이즈 σ","평탄 배경 프레임차 (X)","0.75 / 1.1 / 7.9","2~3","상위 구간 밖"],["피부톤·외형","얼굴 정지 프레임 → LivePortrait (X)","6명 정체성","가상얼굴 6명","정체성 소스로 흡수"]],col_w=[2.2,3.0,3.0,1.6,2.3],size=12)
bullets(s,Inches(0.6),Inches(5.3),Inches(12.1),Inches(2.1),["원칙: 측정 분포를 '맞추는' 것이 아니라 p10~p90을 '포함'하는 범위로 합성(6단계: 실분포 정합 파형이 균등 스윕보다 나빴음; 14단계: 추측한 교란은 해가 됨)","라벨은 학습 분할(1~6)에서만 사용하고 held-out 7~33은 영상·라벨 모두 쓰지 않음 → 비지도 도메인 적응(zero-shot과 구분 표기)","SYN_MMPD 12클립(정체성 6명 × 2)을 CoTAF 풀에 추가한 CoTAF-DM-L을 학습해 CoTAF(zero-shot)·CoTAM-L(실 라벨 in-domain 상한)과 비교"],13)
# 도메인 특화 합성 결과
s=prs.slides.add_slide(blank); n+=1; header(s,"도메인 특화 합성 결과: 범위는 맞췄지만 개선되지 않았다 — 부정 결과와 원인 가설","결과 9 · 도메인 적응 (MMPD · MR-NIRP, 2요인 분리 포함)",n)
pic(s,f"{S}/figH_domain_mmpd.png",Inches(0.6),Inches(1.35),w=Inches(7.3))
table(s,Inches(8.1),Inches(1.4),Inches(4.7),[["MR-NIRP 19명","정지","소동작","대동작","DLCN(부작용)"],["CoTAF (zero-shot)","7.49","9.11","10.78","1.36"],["CoTAF-DN-L (특화 합성)","7.59","7.86","9.25","3.78"],["DM2A-L (실 정지 프레임만)","6.48","8.74","8.23","4.79"],["DM2B-L (가상얼굴+색 통계)","6.66","9.62","10.92","2.72"],["CoTAN-CV-L (실 라벨 CV)","5.69","6.15","9.07","–"],["FUSE","5.27","7.73","7.96","1.01"]],col_w=[1.9,0.65,0.75,0.75,0.65],size=10)
bullets(s,Inches(8.1),Inches(3.3),Inches(4.7),Inches(3.7),["실험의 의미: '대상 데이터셋의 심박·harmonic·조명·동작·센서 통계를 재서 그 범위를 포함하는 합성(12클립, 풀의 5.6%) + 대상 정체성 정지 프레임'이 라벨 없이 in-domain 효과를 낼 수 있는가","결과(같은 cumsum 라벨의 CoTAF-L 12.19 기준): MMPD held-out 13.21 (p=.0098 악화, halving 27). MR-NIRP 소·대동작 평균은 낮아졌으나 n=19 비유의","주의: 이전 판에서 legacy 라벨 CoTAF(11.56, DLCN 1.36)와 비교해 '부작용'으로 읽었던 DLCN 차이는 대부분 기준 오설정(CoTAF-L DLCN 2.42). 융합(FUSE-DM/DN)은 원본 FUSE에 못 미침","2요인 분리 재실험(각 24클립): A=실 정지 프레임만 → 12.44(p=.22, DLCN 4.79 악화) / B=가상얼굴 + 피부 RGB 이식 + 측정 교란 완화 → 11.70(p=.35 차이 없음). 원인은 harmonic·교란이 아니라 압축 정지 프레임 정체성 소스","판정: 채택 안 함. 절차(측정→포함형 합성)는 무해하나 24클립 규모에선 무익; 실 라벨 in-domain(CoTAM-L 9.98)만 유의 개선 → 다음 두 슬라이드: 정체성 규모·실 움직임·해부학 A(x,y)"],11)
note(s,"CoTAF-DM-L = CoTAF 풀 + SYN_MMPD(MMPD 1~6 통계·정체성), CoTAF-DN-L = + SYN_NIRP. 라벨은 학습 분할에서만, held-out 영상·라벨 미사용(비지도 적응). MR-NIRP는 전체 19명 값, in-domain CV는 held-out 폴드 평균. 정본 프로토콜(0.88 BPM 칸).")
# SCALE300 파이프라인
s=prs.slides.add_slide(blank); n+=1; header(s,"외형 축 규모 확대: 피부톤 층화 가상얼굴 203명 × 실 데이터셋 움직임 203종 × 해부학 기반 A(x,y)","방법 강화 2 · 정체성·움직임·공간 구조 (사용자 제안 반영)",n)
pic(s,f"{S}/figK_scale300_pipeline.png",Inches(0.6),Inches(1.4),h=Inches(3.6))
bullets(s,Inches(0.6),Inches(5.1),Inches(12.1),Inches(2.2),["왜: MMPD 학습 분할(1~6)의 피부톤은 3·4뿐인데 held-out 540클립 중 240클립이 피부톤 5·6 — 실 라벨 in-domain(CoTAM-L)조차 Skin 6을 17.0으로 못 고침 → 외형 분포 공백. 기존 코퍼스의 가상얼굴은 9명뿐","정체성: SDXL-Turbo 프롬프트 층화(Fitzpatrick I~VI × 성별 × 연령 × 시드 = 300, ITA 실측 40° → −4° 단조) → LivePortrait 검출 실패 97 제외 → 203명(어두운 피부 V·VI 56명 신규). 왼쪽 그림","움직임: 구동 영상을 5종 → 실 데이터셋 학습 분할 203클립(MMPD 정지/회전/말하기/걷기 120, PURE 6과제 35, MCD 24, UBFC 24; held-out 미사용)으로 확대, LivePortrait가 못 옮기는 병진은 실 얼굴 상자 궤적을 합성기에 직접 이식","공간 구조 A(x,y): 얼굴 파싱으로 이마·볼 1.0 / 코 0.7 / 턱 0.5 / 기타 0.3, 눈·눈썹·입·머리카락·수염 0 + 사람별 랜덤(부위 지터·좌우 비대칭·랜덤필드·고관류 0~40% 탈락). 오른쪽 그림. 기존 uniform은 눈·입까지 맥동, patches는 해부학 무관 반점"],12)
# SCALE300 결과
s=prs.slides.add_slide(blank); n+=1; header(s,"결과: 정체성·움직임 '양'은 전부 해 — 채워야 했던 것은 생체신호의 '형태'였다 (디지털 트윈 → TW 채택)","결과 10 · 16단계 종합 (MMPD held-out 540, 기준 CoTAF-L 12.19)",n)
pic(s,f"{S}/figH_domain_mmpd.png",Inches(0.6),Inches(1.35),h=Inches(4.2))
bullets(s,Inches(7.9),Inches(1.35),Inches(4.9),Inches(5.9),["S300F(고정 주입) 13.35 p=.006 · S300P(피부톤 의존 α·Pbv) 15.63 p<1e-4 · S300R(실 움직임) 14.28 p=.0001 — 전 피부톤 악화, halving 36~47(운동 후 고심박 클립의 high-HR halving)","희석 검정 S300RB(실 클립 ×2) 14.33 ≈ S300R → 원인은 실 비율이 아니라 합성 클립의 내용","S300RA(실 움직임 + A(x,y)) 12.56: 같은 클립에서 A(x,y)만으로 14.28 → 12.56 (p=.0001), 기준선과 동급(p=.17), DLCN 1.85·운전 대동작 7.45 개선 → '어디가 맥동하는가'가 실데이터와 합성의 남은 간극","코퍼스 자체에 A(x,y) 적용(CoTAF-A) 12.88 p=.057 — 이득 없음(A(x,y)는 정체성 클립의 오류 교정용)","+MMPD 실 120 융합(RM/RAM) 10.56/10.58 — CoTAM-L 9.98과 동급, 합성은 실 라벨 위에 아무것도 더하지 못함","MR-NIRP(야간 실차)만은 정체성 합성이 일관 개선(대동작 9.44 → 7.45~8.65, n=19 비유의) — 도메인에 따라 방향이 갈림","→ 디지털 트윈 v1(다음 슬라이드): 실 PPG 파형 뱅크가 유일하게 유의 개선 — CoTAF-TW-L 10.31±0.19 (3시드) 채택"],11)
note(s,"모든 신규 모델은 cumsum 라벨 → 공정 기준은 같은 라벨의 CoTAF-L(12.19). 코퍼스 구성 정정: 실 정체성 161(UBFC 34 + 교란 증강 34 + 압축 10 + PURE 35 + MCD 48) + 가상얼굴 42. Wilcoxon 540 클립 쌍대.")
# 모델 측 개선
s=prs.slides.add_slide(blank); n+=1; header(s,"모델 측 개선: 상보 백본 스펙트럼 융합(FUSE) — 재학습 없이 두 백본의 실패를 서로 지운다","결과 8 · 모델 개선",n)
table(s,Inches(0.6),Inches(1.45),Inches(7.6),[["평가 축","CoTAF","FactorizePhys","FUSE α=0.6"],["MCD 안정 / 운동후","0.48 / 4.17","1.36 / 7.95","0.57 / 3.03"],["PURE 07–10","0.57","0.55","0.53"],["UBFC held-out / +flicker","1.10 / 0.00","2.53 / 30.32","2.53 / 1.43"],["DLCN","1.36","1.01","1.01"],["운전 정지 / 소 / 대","7.49 / 9.11 / 10.78","5.27 / 9.71 / 11.70","5.27 / 7.73 / 7.96"],["MMPD held-out (540)","11.56","14.25","10.53 (p=.009)"]],col_w=[2.3,1.7,1.8,1.8],size=12,hl_cols=(3,))
bullets(s,Inches(8.5),Inches(1.45),Inches(4.3),Inches(5.6),["방법: 두 모델의 정규화 파워 스펙트럼을 P_CoTAF^α · P_FP^(1−α)로 합쳐 argmax. 학습·튜닝 없음(α=0.6 고정, 0.6~0.7 동일 결과)","왜 되는가: 두 백본의 실패 축이 다르다 — PhysFormer+코퍼스는 flicker·harmonic을 이겼고, FactorizePhys는 운동에 강함. 곱은 둘 다 에너지가 있는 주파수(기본파)만 남긴다","합(sum)이나 α=0.5는 flicker에서 FactorizePhys의 lock-in 피크가 이겨 실패(30.32) → CoTAF 우세 가중이 필요","결과: MMPD 유의 개선(p=.009), 운전 3축 전부 개선(대동작 7.96 < POS 9.48, 처음), 비용은 UBFC held-out 1칸(0.88 BPM)","한계: 추론 비용 2배, α를 평가 축에서 고름(민감도 표로 공개). 디코딩 휴리스틱(harmonic 검사·창 중앙값)은 모델별로 효과가 엇갈려 채택 안 함"],13)
# 디지털 트윈 결과
s=prs.slides.add_slide(blank); n+=1; header(s,"디지털 트윈: 측정할 것은 노이즈 통계가 아니라 생체신호의 형태였다 — CoTAF-TW-L 채택","결과 11 · 디지털 트윈 (체인 23~30, 기준 CoTAF-L 12.33±0.29)",n)
table(s,Inches(0.6),Inches(1.45),Inches(7.5),[["실험 (42클립만 교체, 풀 203 동일)","MMPD held-out","판정"],["T: 실 PPG 파형 뱅크 + 정반사 분리 + 코덱 스윕","10.88 (p=.012)","✓ 첫 데이터 측 유의 개선"],["TW: 실 PPG 파형 뱅크만 (UBFC·PURE 학습분할 GT 131 세그먼트)","10.51 (p=.0011) · 3시드 10.31±0.19","★ 채택"],["TS: 정반사 분리만","11.59 (p=.21)","✗ MR-NIRP 후퇴 (야간 NIR에선 하이라이트가 신호)"],["TC: 코덱 스윕만 (mpeg4+H.264 crf23/37)","11.52 (p=.115)","✗ DLCN 4.44 악화"],["TWM: TW + 실 데이터셋 구동 32종","12.88 (TW 대비 p<1e-4)","✗ 움직임 축 닫힘 (5회째 재현)"],["N: MR-NIRP 트윈 (NIRP 통계+정반사 분리)","NIRP 8.8/14.2/11.2","✗"],["전처리: 동적 얼굴 검출","13.36 (p=.013)","✗"],["상한: MMPD 실 라벨 3-fold CV (피부톤 5·6 학습 포함)","9.26 · Skin 6 14.8 · Walking 21.8","실 라벨로도 미해결 → 합성의 목표치"]],col_w=[3.6,2.2,1.7],size=11,hl_rows=(2,))
bullets(s,Inches(8.4),Inches(1.45),Inches(4.3),Inches(5.6),["실험의 의미: 16-6~16-8에서 외형·움직임의 '양'이 전부 실패한 뒤, 합성 '내용의 물리적 정합성'(신호 형태·반사·코덱)을 요인별로 검정","핵심 결과: 해석적 harmonic 파형 → 실 맥파 형태(HRV·dicrotic notch·비대칭)로 바꾸는 것만으로 MMPD −2.0 BPM(시드 분산의 7배), halving 27→12, Walking 24.5→19.6, Skin 5/6·DLCN 동반 개선, 다른 축 무손실","해석: 운동 후 고심박 클립의 high-HR halving은 '모델이 실 맥파의 형태를 모른다'는 문제였다 — 파형 뱅크가 기본파-고조파의 실제 형태 관계를 가르침","정반사·코덱은 MMPD엔 소폭 이득이나 각각 MR-NIRP·DLCN과 상충 → 전 축 zero-shot 기준 채택 불가","실 라벨 상한(9.26)과의 간극은 1.05 BPM; Skin 6·Walking은 상한도 못 풀므로 MMPD 자체의 신호 대 잡음 한계","남은 지렛대(미실험): 움직임-조명 결합·모션 블러 — 실 움직임에서 아직 이식 못 한 성분"],11)
note(s,"파형 뱅크: held-out 미사용(UBFC 1~41·PURE 01~06만), 목표 BPM 리샘플, achieved 라벨 검증. 모든 값 cumsum 라벨·CoTAF-L 기준, Wilcoxon n=540 쌍대. ★=최종 채택.")
# 실험 기록
s=prs.slides.add_slide(blank); n+=1; header(s,"실험 기록: 17단계, 2026-08-21 → 08-31","부록 · 진행 이력",n)
table(s,Inches(0.6),Inches(1.4),Inches(12.1),[["단계","내용","핵심 수치·판정"],["1~2 (08-21)","합성 테스트베드, 노이즈 증강 fine-tuning, 외삽·시드","노이즈 조건 16.32 → 0.13, 시드 std ≤ 0.03"],["3~4 (08-22)","실데이터 co-training, SiNC 손실, 실+통제 노이즈","실+flicker 30.32 → 0.00 (CoTN)"],["5~7 (08-25)","MCD 교차 붕괴 → harmonic 다양화, PURE 저심박, 최종 CoTAF","MCD 42.63 → 0.48, PURE 0.39, 전 축 doubling 0"],["8 (08-25)","DLCN · MR-NIRP 순수 zero-shot","DLCN 1.36 < POS 2.55, 운전 대동작만 열세"],["9 (08-26)","아키텍처 4종 실측, 백본 전이(RFA/FFA), 문헌 비교","적용 가능하나 백본 불변 아님"],["10 (08-26~27)","MMPD 661클립 추가, CoTAM, 학습 라벨 2f 결함 발견·패치","MMPD held-out CoTAF 11.56 / CoTAM-L 9.98 / POS 12.42"],["11 (08-27)","정리 후 전 계보 라벨 수정 재학습(10모델)","효과는 오염 지배 지점 국한, clean-dilution 해석 철회"],["12 (08-27)","UBFC 공정 비교(PURE 가중치 3종, T2L), 발표자료·데모 영상","clean은 모두 쉬움, 통제 교란에서만 갈림"],["13 (08-28)","MR-NIRP in-domain(1~6) + 3-fold CV(12~13명)","CV 5.69 / 6.15 / 9.07 — 대동작은 POS와 동일 → ego-motion 축 필요"],["16 (08-29~30)","모델 측: 디코딩 변형(✗)·FUSE(✓) / 데이터 측 12모델: 통계 특화 DM·DN·2요인, 가상얼굴 203 규모 F/P, 실 움직임 R/RB, 해부학 A(x,y) RA·CoTAF-A, +MMPD 실 RM/RAM","FUSE 10.53(p=.009)·운전 5.3/7.7/8.0 채택; 데이터 측은 CoTAF-L 12.19 대비 유의 개선 없음(RA 12.56 동급, A(x,y)가 유해 절반 상쇄; RB로 희석 가설 기각); 실 라벨 융합만 10.6"],["17 (08-30~31)","디지털 트윈 v1(T)·요인 분해(TW/TS/TC)·NIRP 트윈(N)·동적 검출·MMPD 3-fold CV·다중 시드·구동 다양화(TWM)","TW(실 PPG 파형 뱅크) 채택: 10.31±0.19 vs 12.33±0.29 (3시드); CV 상한 9.26(Skin 6·Walking은 실 라벨로도 미해결); TWM 12.88 → 움직임 축 닫힘"],["15 (08-28)","합성만 학습(68클립) SYNALL-FT / 무작위 초기화 SCR, 전 축 zero-shot","FT: PURE 4.41·운전 8.5/8.2 개선, MCD 31·MMPD 16.5 미해결; SCR 실패 → 합성은 커버리지만 대체"],["14 (08-28)","EGO6: 카메라 ego-motion 합성 6클립 추가(CoTAF3-L)","채택 안 함 — 운전 10.4/10.4/13.5로 악화, MMPD 13.66(p<.001). 아핀 흔들림 ≠ 실차 대동작"]],col_w=[1.5,6.0,4.6],size=12)
# 21 부록
s=prs.slides.add_slide(blank); n+=1; header(s,"부록: 불리한 결과도 그대로 — 재현 규칙","부록",n)
bullets(s,Inches(0.6),Inches(1.45),Inches(12.1),Inches(5.5),["병기한 열세: 운전 대동작(CoTAF 10.78 > POS 9.48), RhythmFormer 전이 시 DLCN·운전 악화, FactorizePhys 불변, CoTAM(원 라벨) 후퇴, CoTAF-L의 DLCN·MMPD 소폭 후퇴","in-domain과 zero-shot을 표에서 분리 표기. 평가 프로토콜 계열 혼용 금지(동일 모델이 0.98 vs 7.02까지 벌어진 문헌 사례)","모든 수치의 정본: _workspace/16_physformer_experiment.md, results/tables_v2.md(T1~T6), stats_*.json — 자동 재생성 스크립트로 검증 가능","toolbox 로컬 패치: RESUME 로드, HR 클램프, RPPG_HR_LABEL=cumsum, RhythmFormer 손실 클램프, FactorizePhys strict=False, numpy2, 전처리 병렬 수, BlandAltman KDE","재현: conda env vface-rppg(cu128), yaml5~19, GPU 게이트 체인 스크립트(공유 머신), 캐시 즉시 삭제 정책","온보딩 문서 8종: Desktop/Virtual-Face/onboarding/ (연구개요 · 환경 · 데이터셋 · 파이프라인 · 결과 · 논문 · 규칙)"],15)
prs.save(OUT); print("saved",OUT,"slides",n)
