import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import re
import math
from typing import List, Dict, Optional
from skimage import color # requirements.txt에 'scikit-image'가 있어야 합니다.
import matplotlib.pyplot as plt
from datetime import datetime # DATE 표시를 위해 import
from io import BytesIO # 엑셀 다운로드를 위해 import
import sys, types

# ==========================================================
# 0. CONFIG (Enhanced Version)
# ==========================================================
CONFIG = {
    'embed_dim': 64, # ⭐️ 모델 뼈대 생성을 위해 필수

    # 필수 컬럼 매핑
    'condition_col': 'COLOR',
    'name_col':      'COLOR',
    'lab_cols':      ['L*(10°/D65)', 'a*(10°/D65)', 'b*(10°/D65)'],
    'total_col' : "TOTAL_LOAD",

    # 스펙트럼 사용 (수동 지정)
    'spectrum_prefixes': [],
    'spectrum_cols':   ['400[nm]', '410[nm]', '420[nm]', '430[nm]', '440[nm]', '450[nm]',
       '460[nm]', '470[nm]', '480[nm]', '490[nm]', '500[nm]', '510[nm]',
       '520[nm]', '530[nm]', '540[nm]', '550[nm]', '560[nm]', '570[nm]',
       '580[nm]', '590[nm]', '600[nm]', '610[nm]', '620[nm]', '630[nm]',
       '640[nm]', '650[nm]', '660[nm]', '670[nm]', '680[nm]', '690[nm]',
       '700[nm]'],

    # 레시피(56 안료)
    'recipe_cols': [
       '1/10 BLUE 2000/S100', '1/10 BROWN 3001/S100', '1/10 CARBON',
       '1/10 CARBON/S100', '1/10 CO BLUE/R350', '1/10 CO BLUE/S100',
       '1/10 GREEN K8730/S100', '1/10 MK4535/R350', '1/10 MK4535/S100',
       '1/10 RED B/S100', '1/10 YELLOW300/S100', '1/100 BLUE7000/S100',
       '1/100 CARBON', '1/100 CARBON/S100', '1/100 MK4535/S100',
       '1/100 RED B/R350', '1/100 RED B/S100', '1/100 YELLOW 300/S100',
       '1/50 CARBON/S100', '1/5000 CARBON/S100', '10550 BROWN', '2000 BLUE',
       '214 BLUE', '23 VIOLET', '7000 BLUE', 'BLUE 424', 'BROWN 216',
       'BROWN 3001', 'CARBON', 'CO BLUE', 'GREEN 9361', 'GREEN K8730',
       'HQ BLUE', 'HQ GREEN', 'HQ MAGENTA', 'HQ ORANGE', 'HQ ORANGE+RED',
       'HQ ORANGE+YELLOW', 'HQ PINK', 'HQ RED', 'HQ VIOLET', 'HQ YELLOW',
       'MK 4535', 'ORANGE K2890(2G)', 'ORANGE K2960', 'RED B', 'RED BNP',
       'RED K3840', 'RED K4035(2B)', 'TIO2-R350', 'VIOLET 21', 'VIOLET 42',
       'YELLOW 10401', 'YELLOW 300', 'YELLOW H3R', 'YELLOW NG'
    ],

    "tio2_name": "TIO2-R350",

    # Enhanced pipeline parameters
    'd_model': 128,
    'nhead': 4,
    'nlayers': 2,
    'lr': 0.002,
    'surrogate_loss_weight': 0.1,
    'use_similar_init': True,
    'n_similar': 3,  # Number of similar recipes to blend
    'top_k': 5,  # Number of top pigments to select
    'model_weight': 0.7,  # Weight for model prediction (vs similar recipes)
    'n_iterations': 10,  # Surrogate refinement iterations
}


# ==========================================================
# 1. 텍스트 인코더 클래스 정의 (SimpleNameEncoder)
# ==========================================================
class SimpleNameEncoder:
    """
    Simple token-based text encoder for color names
    """
    def __init__(self, max_tokens=512, embed_dim=64):
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.vocab = {}
        self.inv_vocab = {}

    def fit(self, texts):
        """Build vocabulary from texts"""
        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
        from collections import Counter
        token_counts = Counter(all_tokens)
        top_tokens = [t for t, _ in token_counts.most_common(self.max_tokens - 2)]
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, token in enumerate(top_tokens, start=2):
            self.vocab[token] = i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def encode(self, text):
        """
        Encode text to one-hot style embedding
        Returns: (embed_dim,) numpy array
        """
        tokens = self._tokenize(text)
        token_ids = [self.vocab.get(t, 1) for t in tokens]
        embedding = np.zeros(self.embed_dim)
        for tid in token_ids:
            if tid < self.embed_dim:
                embedding[tid] += 1
        if embedding.sum() > 0:
            embedding = embedding / embedding.sum()
        return embedding


# ==========================================================
# 2. PyTorch 모델 클래스 정의 (CrossAttentionBlock, RecipeNet3Head)
# ==========================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
    def forward(self, x, context):
        h, _ = self.attn(x, context, context)
        x = self.norm1(x + h)
        h = self.ffn(x)
        x = self.norm2(x + h)
        return x


class RecipeNet3Head(nn.Module):
    def __init__(self, in_dim, num_pigments, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.num_pigments = num_pigments
        self.d_model = d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pigment_embed = nn.Parameter(torch.randn(num_pigments, d_model))
        self.attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead) for _ in range(nlayers)
        ])
        self.base_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.chroma_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_pigments - 1)
        )
        self.total_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
    def forward(self, x):
        B = x.size(0)
        q = self.input_proj(x).unsqueeze(1)
        context = self.pigment_embed.unsqueeze(0).expand(B, -1, -1)
        for layer in self.attn_layers:
            q = layer(q, context)
        q = q.squeeze(1)
        b = self.base_head(q)
        chroma = torch.softmax(self.chroma_head(q), dim=-1)
        total = self.total_head(q)
        return b, chroma, total

# ==========================================================
# 3. 유틸리티 함수 (DeltaE, 색상 시각화, 엑셀 변환)
# ==========================================================
def lab_to_rgb(lab):
    """Lab -> RGB 변환 (skimage 활용)"""
    lab = np.array(lab).reshape(1,1,3)
    rgb = color.lab2rgb(lab)
    return rgb[0,0,:]

def show_color_patches(lab_true, lab_pred):
    """Streamlit용 색상 비교차트 생성 (True vs Pred)"""
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    rgb_true = lab_to_rgb(lab_true)
    rgb_pred = lab_to_rgb(lab_pred)
    ax[0].imshow([[rgb_true]]); ax[0].set_title("Target (True)"); ax[0].axis("off")
    ax[1].imshow([[rgb_pred]]); ax[1].set_title("Predicted (Surrogate)"); ax[1].axis("off")
    return fig

def show_single_color_patch(lab_color, title="Color"):
    """Streamlit용 단일 색상 차트 생성"""
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    rgb_color = lab_to_rgb(lab_color)
    ax.imshow([[rgb_color]])
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return fig

def deltaE_00(y_true, y_pred, kL=1, kC=1, kH=1):
    """CIEDE2000 DeltaE 계산"""
    L1, a1, b1 = y_true[:,0], y_true[:,1], y_true[:,2]
    L2, a2, b2 = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    C1 = np.sqrt(a1*a1 + b1*b1); C2 = np.sqrt(a2*a2 + b2*b2)
    C_bar = 0.5 * (C1 + C2); C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7 + 1e-12)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1); C2p = np.sqrt(a2p*a2p + b2*b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0; h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(C1p*C2p==0,0.0,dhp); dhp = np.where(dhp>180,dhp-360,dhp); dhp = np.where(dhp<-180,dhp+360,dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2.0)
    Lp_bar = 0.5*(L1+L2); Cp_bar=0.5*(C1p+C2p)
    hp_sum = h1p+h2p
    hp_bar = np.where((C1p*C2p)==0,hp_sum, np.where(np.abs(h1p-h2p)>180,(hp_sum+360.0)/2.0-360.0*(hp_sum>=360.0),hp_sum/2.0))
    T=(1-0.17*np.cos(np.radians(hp_bar-30)) +0.24*np.cos(np.radians(2*hp_bar)) +0.32*np.cos(np.radians(3*hp_bar+6)) -0.20*np.cos(np.radians(4*hp_bar-63)))
    Sl=1+0.015*(Lp_bar-50)**2/np.sqrt(20+(Lp_bar-50)**2)
    Sc=1+0.045*Cp_bar; Sh=1+0.015*Cp_bar*T
    delta_theta=30*np.exp(-((hp_bar-275)/25)**2)
    Rc=2*np.sqrt(C_bar**7/(C_bar**7+25**7+1e-12))
    Rt=-Rc*np.sin(2*np.radians(delta_theta))
    dE00=np.sqrt((dLp/(kL*Sl))**2+(dCp/(kC*Sc))**2+(dHp/(kH*Sh))**2 +Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE00

# 엑셀 다운로드를 위한 헬퍼 함수
def to_excel_with_header(recipe_df, color_name, date_str):
    """Pandas DataFrame(레시피)과 헤더 정보(색상명, 날짜)를 엑셀 파일(BytesIO)로 변환합니다."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 헤더 정보 DataFrame 생성
        header_data = {'Info': ['COLOR', 'DATE'], 'Value': [color_name, date_str]}
        header_df = pd.DataFrame(header_data)
        # 헤더 DataFrame 쓰기 (인덱스 및 헤더 제외)
        header_df.to_excel(writer, index=False, header=False, sheet_name='Predicted_Recipe', startrow=0)

        # 2. 레시피 DataFrame 쓰기 (헤더 포함, 헤더 정보 아래에 위치)
        recipe_df.to_excel(writer, index=False, header=True, sheet_name='Predicted_Recipe', startrow=3)

        # (선택사항) 컬럼 너비 자동 조절
        worksheet = writer.sheets['Predicted_Recipe']
        for idx, col in enumerate(recipe_df): # recipe_df의 컬럼 기준
            series = recipe_df[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
            )) + 2 # 약간의 여유 공간
            worksheet.column_dimensions[chr(65 + idx)].width = max_len # A열부터 시작

    processed_data = output.getvalue()
    return processed_data

# ==========================================================
# 4. Enhanced Helper Functions (NEW in v2)
# ==========================================================
def find_top_k_similar_recipes(lab_query, df_reference, lab_cols, recipe_cols, top_k=3, epsilon=1e-5):
    """
    Find top k similar recipes based on Lab distance and create weighted average
    """
    lab_ref = df_reference[lab_cols].values
    distances = np.sqrt(((lab_ref - lab_query.reshape(1, -1))**2).sum(axis=1))
    top_k_indices = np.argsort(distances)[:top_k]
    top_k_distances = distances[top_k_indices]

    # Distance-based weights: w_i = 1 / (d_i + epsilon)
    weights = 1.0 / (top_k_distances + epsilon)
    weights = weights / weights.sum()

    top_k_info = []
    recipes = []

    for i, idx in enumerate(top_k_indices):
        recipe = df_reference.iloc[idx][recipe_cols].values.astype(np.float32)
        color_name = df_reference.iloc[idx].get('COLOR', f'Sample_{idx}')
        lab_value = df_reference.iloc[idx][lab_cols].values

        top_k_info.append({
            'rank': i + 1,
            'index': idx,
            'color_name': color_name,
            'distance': top_k_distances[i],
            'weight': weights[i],
            'lab': lab_value,
            'recipe': recipe
        })
        recipes.append(recipe)

    # Weighted average recipe
    recipes = np.array(recipes)
    weighted_recipe = np.zeros(len(recipe_cols), dtype=np.float32)
    for i, w in enumerate(weights):
        weighted_recipe += w * recipes[i]

    return weighted_recipe, top_k_info


def select_top_k_pigments(recipe_array, recipe_cols, top_k=5):
    """Select top k pigments and set rest to 0"""
    result = np.zeros_like(recipe_array)
    top_k_indices = np.argsort(recipe_array)[-top_k:][::-1]

    for idx in top_k_indices:
        if recipe_array[idx] > 0:
            result[idx] = recipe_array[idx]

    selected_pigments = []
    for idx in top_k_indices:
        if recipe_array[idx] > 0:
            selected_pigments.append({
                'name': recipe_cols[idx],
                'amount': recipe_array[idx],
                'index': idx
            })

    return result, selected_pigments


def normalize_recipe_array(recipe_array, total_load):
    """Normalize recipe and apply total load"""
    total = recipe_array.sum()
    if total > 0:
        normalized = (recipe_array / total) * total_load
        return normalized
    return recipe_array

# ==========================================================
# 5. Enhanced Inference Function (NEW)
# ==========================================================
def run_inference_enhanced(model, cfg, surrogate, spectrum, lab, color_name, name_encoder, df_reference):
    """
    Enhanced recipe prediction pipeline with:
    1. Similar recipe weighted initialization (top n_similar)
    2. Model prediction + similar recipe blending (model_weight ratio)
    3. Top-k pigment selection
    4. Surrogate-based color difference calculation
    """
    device = torch.device("cpu") # Streamlit Cloud는 CPU 기반
    model = model.to(device)
    model.eval()

    # ---- 텍스트 임베딩
    X_text = name_encoder.encode(color_name) # shape (1, embed_dim)

    # ---- 입력 feature
    feat = np.hstack([spectrum, lab, X_text])
    xb = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)

    # === STEP 1: Similar recipe initialization ===
    initial_recipe = None
    similar_info = None
    if cfg['use_similar_init'] and df_reference is not None:
        weighted_recipe, top_k_info = find_top_k_similar_recipes(
            lab, df_reference, cfg['lab_cols'], cfg['recipe_cols'],
            top_k=cfg['n_similar'], epsilon=1e-5
        )
        initial_recipe = weighted_recipe
        similar_info = top_k_info

    # === STEP 2: Model prediction ===
    with torch.no_grad():
        b, q, t = model(xb)
        base_idx = cfg['recipe_cols'].index(cfg['tio2_name'])
        others = (1-b)*q
        chunks = []
        k = 0
        for j in range(len(cfg['recipe_cols'])):
            if j == base_idx:
                chunks.append(b)
            else:
                chunks.append(others[:,k:k+1])
                k += 1
        p = torch.cat(chunks, dim=1)

        # === STEP 3: Blend model + similar recipes ===
        if initial_recipe is not None:
            p_model = p.cpu().numpy().flatten()
            p_mixed = cfg['model_weight'] * p_model + (1 - cfg['model_weight']) * initial_recipe
            p_mixed = p_mixed / p_mixed.sum()  # Re-normalize
            p = torch.from_numpy(p_mixed.astype(np.float32)).unsqueeze(0).to(device)

        total_load = t.cpu().numpy().flatten()[0]
        P_g_raw = (p * t).cpu().numpy().flatten()

    # === STEP 4: Top-k pigment selection ===
    P_g_topk, selected_pigments = select_top_k_pigments(
        P_g_raw, cfg['recipe_cols'], top_k=cfg['top_k']
    )
    P_g_topk = normalize_recipe_array(P_g_topk, total_load)

    # === STEP 5: Surrogate refinement (optional) ===
    if cfg['n_iterations'] > 0:
        # Set random seed for reproducibility
        np.random.seed(42)

        P_g_refined = P_g_topk.copy()
        selected_indices = [p['index'] for p in selected_pigments]

        best_de = float('inf')
        best_recipe = P_g_refined.copy()

        for iter in range(cfg['n_iterations']):
            lab_pred_current = surrogate.predict(P_g_refined.reshape(1, -1))
            de_current = deltaE_00(lab.reshape(1, 3), lab_pred_current).mean()

            if de_current < best_de:
                best_de = de_current
                best_recipe = P_g_refined.copy()

            # Add noise to selected pigments only
            noise = np.zeros_like(P_g_refined)
            for idx in selected_indices:
                noise[idx] = np.random.normal(0, 0.05)

            P_g_candidate = np.clip(P_g_refined * (1 + noise), 0, None)
            P_g_candidate = normalize_recipe_array(P_g_candidate, total_load)

            lab_pred_candidate = surrogate.predict(P_g_candidate.reshape(1, -1))
            de_candidate = deltaE_00(lab.reshape(1, 3), lab_pred_candidate).mean()

            if de_candidate < de_current:
                P_g_refined = P_g_candidate

        P_g_topk = best_recipe if best_de < de_current else P_g_refined

    # === STEP 6: Surrogate prediction ===
    lab_pred = surrogate.predict(P_g_topk.reshape(1, -1)).flatten()
    de00 = deltaE_00(lab.reshape(1, 3), lab_pred.reshape(1, 3)).mean()

    # ---- Streamlit 출력 ----

    st.subheader("🔬 예측된 레시피 (Enhanced Pipeline)")

    # --- 테이블 1: 정보 (COLOR, DATE 통합) ---
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.markdown(f"""
    <style>
        .info-table {{ border-collapse: collapse; width: 60%; margin-bottom: 1rem; }}
        .info-table td, .info-table th {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .info-table th {{ background-color: #f2f2f2; font-weight: bold; }}
    </style>
    <table class="info-table">
      <tr>
        <th>COLOR</th>
        <td>{color_name}</td>
        <th>DATE</th>
        <td>{current_date}</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.write("") # 약간의 간격

    # --- 유사 레시피 정보 표시 (NEW) ---
    if similar_info:
        with st.expander("📊 유사 레시피 참조 정보 (Similar Recipes Used)"):
            similar_df = pd.DataFrame([
                {
                    'Rank': info['rank'],
                    'Color Name': info['color_name'],
                    'Distance (ΔE)': f"{info['distance']:.2f}",
                    'Weight': f"{info['weight']:.3f}"
                }
                for info in similar_info
            ])
            st.dataframe(similar_df, hide_index=True, use_container_width=True)
            st.caption(f"🔍 모델 예측 {cfg['model_weight']*100:.0f}% + 유사 레시피 {(1-cfg['model_weight'])*100:.0f}% 혼합")

    # --- 테이블 2: 안료 (PIGMENT, 함량) ---
    recipe_g_series = pd.Series(P_g_topk, index=cfg['recipe_cols'])

    # 1. 0.01 이상 필터링 & 내림차순 정렬
    recipe_filtered = recipe_g_series[recipe_g_series >= 0.01].sort_values(ascending=False)

    # 2. 표시할 레시피 결정 (상위 6개 또는 전체)
    if len(recipe_filtered) > 6:
        recipe_to_display_series = recipe_filtered.head(6)
    else:
        recipe_to_display_series = recipe_filtered

    # 3. 화면 표시 및 다운로드용 데이터 준비
    if recipe_to_display_series.empty:
        st.warning("예측된 레시피 중 함량이 0.01 g/K 이상인 안료가 없습니다.")
        recipe_df_final = pd.DataFrame({'PIGMENT': [], '함량 (g/K)': []})
    else:
        # DataFrame으로 변환 (화면 표시 및 다운로드 공통 사용)
        recipe_df_final = pd.DataFrame({
            'PIGMENT': recipe_to_display_series.index,
            '함량 (g/K)': recipe_to_display_series.values
        }).reset_index(drop=True)

        # 소수점 4자리까지만 화면에 표시
        st.dataframe(
            recipe_df_final.style.format({'함량 (g/K)': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )

    # 엑셀 다운로드 버튼
    excel_data = to_excel_with_header(recipe_df_final, color_name, current_date)
    st.download_button(
        label="📄 표시된 레시피 엑셀 다운로드 (.xlsx)",
        data=excel_data,
        file_name=f'predicted_recipe_{color_name.replace(" ", "_")}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    st.divider() # 가로줄 추가

    # ⭐ Save predicted recipe to session state for simulator (do NOT auto-load)
    st.session_state.last_predicted_recipe = P_g_topk
    st.session_state.last_predicted_lab = lab.flatten()

    # Guide user to simulator
    st.info("💡 **아래 '레시피 시뮬레이터'에서 예측된 레시피를 불러와 색상을 확인하고 조정할 수 있습니다!**")


# ==========================================================
# 6. 모델 로드 (Streamlit 캐시 사용)
# ==========================================================
@st.cache_resource
def load_all_models(config):
    device = torch.device("cpu")
    
    # 학습 코드의 models.SimpleNameEncoder를 로컬 클래스로 매핑
    _models_mod = types.ModuleType('models')
    _models_mod.SimpleNameEncoder = SimpleNameEncoder
    sys.modules['models'] = _models_mod
    
    try: name_encoder = joblib.load("name_encoder.pkl")
    except FileNotFoundError: st.error("`name_encoder.pkl` 파일을 찾을 수 없습니다."); return None, None, None
    try: surrogate = joblib.load("xgb_surrogate_3.pkl")
    except FileNotFoundError: st.error("`xgb_surrogate_3.pkl` 파일을 찾을 수 없습니다."); return None, None, None
    try:
        in_dim = len(config['spectrum_cols']) + len(config['lab_cols']) + config['embed_dim']
        num_pigments = len(config['recipe_cols'])
        model = RecipeNet3Head(
            in_dim,
            num_pigments,
            d_model=config['d_model'],
            nhead=config['nhead'],
            nlayers=config['nlayers']
        )
        model.load_state_dict(torch.load("recipe_model_optimized_weight_0.1.pth", map_location=device))
        model.eval()
    except FileNotFoundError: st.error("`recipe_model_optimized_weight_0.1.pth` 파일을 찾을 수 없습니다."); return None, None, None
    except Exception as e: st.error(f"PyTorch 모델 로드 오류: {e}"); return None, None, None
    return model, name_encoder, surrogate

@st.cache_data
def load_reference_data():
    """Load reference dataset for similar recipe initialization"""
    try:
        df = pd.read_csv("swatch_recipe_merged_1120.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ 참조 데이터셋 (swatch_recipe_merged_1120.csv)을 찾을 수 없습니다. 유사 레시피 기능이 비활성화됩니다.")
        return None
    except Exception as e:
        st.error(f"참조 데이터셋 로드 오류: {e}")
        return None

# ==========================================================
# 7. Streamlit UI (메인 앱 로직)
# ==========================================================

# --- 엑셀 파일 처리 함수 ---
@st.cache_data
def parse_excel(uploaded_file, config):
    try:
        df = pd.read_excel(uploaded_file)
        filter_col = '정반사광 처리'; name_col = '데이터 이름'
        if filter_col not in df.columns: st.error(f"'{filter_col}' 없음"); return None
        sce_df = df[df[filter_col] == 'SCE'].copy()
        if sce_df.empty: st.error("'SCE' 행 없음"); return None
        if name_col not in sce_df.columns: st.error(f"'{name_col}' 없음"); return None
        sce_df['Color Name'] = sce_df[name_col].astype(str).str[4:].str.strip()
        required_cols = config['lab_cols'] + config['spectrum_cols']
        missing_cols = [col for col in required_cols if col not in sce_df.columns]
        if missing_cols: st.error(f"필수 컬럼 없음: {missing_cols}"); return None
        final_cols = ['Color Name'] + config['lab_cols'] + config['spectrum_cols']
        for col in config['lab_cols'] + config['spectrum_cols']:
             sce_df[col] = pd.to_numeric(sce_df[col], errors='coerce')
        if sce_df[final_cols].isnull().values.any():
            nan_rows = sce_df[sce_df[final_cols].isnull().any(axis=1)]['Color Name'].tolist()
            st.warning(f"숫자 오류 행 제외: {nan_rows}")
            sce_df = sce_df.dropna(subset=final_cols)
        if sce_df.empty: st.error("유효 'SCE' 행 없음"); return None
        return sce_df[final_cols].reset_index(drop=True)
    except Exception as e: st.error(f"엑셀 처리 오류: {e}"); return None

# --- 메인 UI ---
st.set_page_config(layout="wide")
st.title("🧪 레시피 예측 모델 v2 (Enhanced)")
st.caption("✨ Similar Recipe Blending + Top-K Selection + Surrogate Refinement")

# 모델 로드
model, name_encoder, surrogate = load_all_models(CONFIG)
df_reference = load_reference_data()

if model and name_encoder and surrogate:
    st.header("1. 목표 색상 정보 업로드")

    uploaded_file = st.file_uploader(
        "목표 색상 엑셀 파일 업로드 (xlsx)",
        type=["xlsx"],
        help="파일 내 '정반사광 처리' 컬럼의 'SCE' 행 데이터를 모두 불러옵니다."
    )

    # 세션 상태 초기화
    if 'sce_data' not in st.session_state: st.session_state.sce_data = None
    if 'selected_color' not in st.session_state: st.session_state.selected_color = None
    if 'prediction_output' not in st.session_state: st.session_state.prediction_output = None

    # 파일 업로드 시 처리
    if uploaded_file is not None:
        new_sce_data = parse_excel(uploaded_file, CONFIG)
        if new_sce_data is not None and not new_sce_data.empty:
            st.session_state.sce_data = new_sce_data
            # Selectbox의 기본값을 첫 번째 항목으로 설정
            st.session_state.selected_color = st.session_state.sce_data['Color Name'][0]
            # 새 파일 업로드 시 이전 예측 결과 삭제
            if 'prediction_output' in st.session_state: del st.session_state.prediction_output
        else: # 파싱 실패 시 초기화
             st.session_state.sce_data = None; st.session_state.selected_color = None
             if 'prediction_output' in st.session_state: del st.session_state.prediction_output

    # --- 데이터 선택 및 표시 ---
    if st.session_state.sce_data is not None:
        df_sce = st.session_state.sce_data
        if not df_sce.empty:
            st.header("2. 목표 색상 선택")
            selected_color_name_from_box = st.selectbox(
                f"'SCE' 기준 총 {len(df_sce)}개의 색상이 로드되었습니다. 예측할 색상을 선택하세요",
                options=df_sce['Color Name'], key='color_selector',
                index=list(df_sce['Color Name']).index(st.session_state.selected_color) if st.session_state.selected_color in list(df_sce['Color Name']) else 0
            )
            # Selectbox 변경 시 세션 상태 업데이트 및 예측 결과 초기화
            if st.session_state.selected_color != selected_color_name_from_box:
                 st.session_state.selected_color = selected_color_name_from_box
                 if 'prediction_output' in st.session_state: del st.session_state.prediction_output

            current_selected_color = st.session_state.selected_color
            if current_selected_color and current_selected_color in list(df_sce['Color Name']):
                selected_row = df_sce[df_sce['Color Name'] == current_selected_color].iloc[0]
                st.subheader(f"'{current_selected_color}' 데이터 확인")
                lab_true_np = selected_row[CONFIG['lab_cols']].values.astype(float)
                spectrum_true_np = selected_row[CONFIG['spectrum_cols']].values.astype(float)
                col1, col2, col3 = st.columns([0.45, 0.4, 0.15])
                with col1: # Lab 정보
                    st.write("**목표 색상 정보:**")
                    st.text_input("Color Name", value=current_selected_color, disabled=True, key=f"name_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][0]}", value=f"{lab_true_np[0]:.2f}", disabled=True, key=f"l_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][1]}", value=f"{lab_true_np[1]:.2f}", disabled=True, key=f"a_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][2]}", value=f"{lab_true_np[2]:.2f}", disabled=True, key=f"b_display_{current_selected_color}")
                with col2: # 스펙트럼 정보
                    st.write("**스펙트럼 정보:**")
                    spectrum_df = pd.DataFrame({'파장 (Wavelength)': CONFIG['spectrum_cols'], '값 (Value)': spectrum_true_np})
                    st.dataframe(spectrum_df, height=320)
                with col3: # 색상 시각화
                    st.write("**Target Color:**")
                    fig = show_single_color_patch(lab_true_np, title="Target (True)")
                    st.pyplot(fig)


                # --- 예측 버튼 ---
                st.header("3. 예측 실행")
                if st.button(f"🚀 '{current_selected_color}' 레시피 예측 실행 (Enhanced)", type="primary", key=f"predict_btn_{current_selected_color}"):
                    with st.spinner('Enhanced 모델 예측 중...'):
                        st.session_state.prediction_output = {
                             "model": model, "cfg": CONFIG, "surrogate": surrogate,
                             "spectrum": spectrum_true_np, "lab": lab_true_np,
                             "color_name": current_selected_color, "name_encoder": name_encoder,
                             "df_reference": df_reference
                        }

    # --- 파일 없음 또는 초기화 ---
    elif uploaded_file is None:
        st.info("⬆️ 예측을 시작하려면 엑셀 파일을 업로드해주세요.")
        if 'sce_data' in st.session_state: del st.session_state.sce_data
        if 'selected_color' in st.session_state: del st.session_state.selected_color
        if 'prediction_output' in st.session_state: del st.session_state.prediction_output

    # --- 예측 결과 표시 ---
    if 'prediction_output' in st.session_state and st.session_state.prediction_output is not None:
         if st.session_state.selected_color == st.session_state.prediction_output['color_name']:
              output_args = st.session_state.prediction_output
              run_inference_enhanced(**output_args)

    # ==========================================================
    # NEW FEATURE: Recipe Simulator (역방향 시뮬레이터)
    # ==========================================================
    st.divider()
    st.header("🔬 레시피 시뮬레이터 (Recipe → Color Prediction)")
    st.caption("레시피를 직접 조정하면서 예상 색상과 Target과의 색차(ΔE)를 실시간으로 확인하세요")

    # Load surrogate model for simulation
    @st.cache_resource
    def load_simulator_surrogate():
        try:
            surrogate_sim = joblib.load("xgb_surrogate_3.pkl")
            return surrogate_sim
        except FileNotFoundError:
            st.error("⚠️ xgb_surrogate_3.pkl 파일을 찾을 수 없습니다. 시뮬레이터 기능이 비활성화됩니다.")
            return None

    surrogate_simulator = load_simulator_surrogate()

    if surrogate_simulator:
        # Initialize session state for simulator
        if 'simulator_recipe' not in st.session_state:
            st.session_state.simulator_recipe = {col: 0.0 for col in CONFIG['recipe_cols']}
        if 'target_lab_sim' not in st.session_state:
            st.session_state.target_lab_sim = None
        if 'recipe_reload_timestamp' not in st.session_state:
            st.session_state.recipe_reload_timestamp = 0
        if 'simulator_result' not in st.session_state:
            st.session_state.simulator_result = None

        # Recipe input section
        st.subheader("1️⃣ 레시피 입력/조정")

        col_recipe1, col_recipe2 = st.columns([0.7, 0.3])

        with col_recipe1:
            # Quick load from prediction
            if 'last_predicted_recipe' in st.session_state and st.session_state.last_predicted_recipe is not None:
                st.info("💡 위에서 예측한 레시피를 불러올 수 있습니다. 버튼을 눌러 레시피를 로드하세요!")
                if st.button("🔄 예측 레시피 불러오기", key="reload_predicted_recipe"):
                    # Reload predicted recipe into simulator
                    predicted_recipe_array = st.session_state.last_predicted_recipe
                    st.session_state.simulator_recipe = {
                        col: float(predicted_recipe_array[idx])
                        for idx, col in enumerate(CONFIG['recipe_cols'])
                    }
                    # Also set target Lab if available
                    if 'last_predicted_lab' in st.session_state:
                        st.session_state.target_lab_sim = st.session_state.last_predicted_lab
                    # Update timestamp to force widget refresh
                    import time
                    st.session_state.recipe_reload_timestamp = time.time()
                    st.success("✅ 예측된 레시피를 불러왔습니다!")
                    st.rerun()
            else:
                st.warning("⚠️ 먼저 위에서 레시피 예측을 실행해주세요.")

            # Recipe input - Top-N only
            num_pigments = st.slider("안료 개수", min_value=1, max_value=10, value=5, key="num_pigments_sim")

            # Get top N pigments from current simulator recipe
            recipe_series = pd.Series(st.session_state.simulator_recipe)
            top_pigments_data = recipe_series[recipe_series > 0].sort_values(ascending=False).head(num_pigments)
            top_pigments_list = top_pigments_data.index.tolist()
            top_amounts_list = top_pigments_data.values.tolist()

            # Pad with empty entries if needed
            while len(top_pigments_list) < num_pigments:
                top_pigments_list.append("")
                top_amounts_list.append(0.0)

            st.write("**안료 선택 및 함량 입력:**")

            selected_pigments = []
            for i in range(num_pigments):
                # Get default pigment and amount from top_pigments_list
                default_pigment = top_pigments_list[i] if i < len(top_pigments_list) else ""

                col_pig, col_amt = st.columns([0.6, 0.4])
                with col_pig:
                    # Set default index based on top pigments
                    if default_pigment and default_pigment in CONFIG['recipe_cols']:
                        default_index = CONFIG['recipe_cols'].index(default_pigment) + 1  # +1 because of "" at index 0
                    else:
                        default_index = 0

                    pigment = st.selectbox(
                        f"안료 {i+1}",
                        options=[""] + CONFIG['recipe_cols'],
                        index=default_index,
                        key=f"pigment_select_{i}_{st.session_state.recipe_reload_timestamp}"
                    )
                with col_amt:
                    # Show amount - use the value from simulator_recipe if pigment is selected
                    if pigment:
                        default_amount = st.session_state.simulator_recipe.get(pigment, 0.0)
                    else:
                        default_amount = 0.0

                    amount = st.number_input(
                        f"함량 (g/K)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_amount),
                        step=0.01,
                        format="%.4f",
                        key=f"pigment_amount_{i}_{st.session_state.recipe_reload_timestamp}"
                    )
                    if pigment:
                        selected_pigments.append((pigment, amount))

            # Reset all to 0, then set selected ones
            if st.button("✅ 레시피 적용", key="apply_recipe_topn"):
                st.session_state.simulator_recipe = {col: 0.0 for col in CONFIG['recipe_cols']}
                for pigment, amount in selected_pigments:
                    if pigment:
                        st.session_state.simulator_recipe[pigment] = amount
                st.success("레시피가 적용되었습니다!")
                st.rerun()

        with col_recipe2:
            st.write("**현재 레시피 요약:**")
            current_recipe_array = np.array([st.session_state.simulator_recipe[col] for col in CONFIG['recipe_cols']])
            total_amount = current_recipe_array.sum()
            non_zero = (current_recipe_array > 0).sum()

            st.metric("총 함량", f"{total_amount:.2f} g/K")
            st.metric("사용 안료 수", f"{non_zero}개")

            if st.button("🔄 레시피 초기화", key="reset_recipe"):
                st.session_state.simulator_recipe = {col: 0.0 for col in CONFIG['recipe_cols']}
                st.success("레시피가 초기화되었습니다!")
                st.rerun()

        st.divider()

        # Simulation and visualization
        st.subheader("2️⃣ 색상 예측 결과")

        if st.button("🎨 색상 예측 실행", type="primary", key="run_simulation"):
            if st.session_state.target_lab_sim is None:
                st.error("Target 색상을 먼저 설정해주세요!")
            elif total_amount == 0:
                st.error("레시피를 입력해주세요!")
            else:
                with st.spinner('색상 예측 중...'):
                    # Predict color from recipe
                    recipe_array = np.array([st.session_state.simulator_recipe[col] for col in CONFIG['recipe_cols']]).reshape(1, -1)
                    predicted_lab = surrogate_simulator.predict(recipe_array).flatten()

                    # Calculate delta E
                    target_lab = st.session_state.target_lab_sim.reshape(1, 3)
                    pred_lab = predicted_lab.reshape(1, 3)
                    delta_e = deltaE_00(target_lab, pred_lab).mean()

                    # Save to session state
                    st.session_state.simulator_result = {
                        'delta_e': delta_e,
                        'predicted_lab': predicted_lab,
                        'target_lab': target_lab,
                        'recipe_array': recipe_array
                    }

        # Display results from session state (persists across reruns)
        if st.session_state.simulator_result is not None:
            result = st.session_state.simulator_result
            delta_e = result['delta_e']
            predicted_lab = result['predicted_lab']
            target_lab = result['target_lab']
            recipe_array = result['recipe_array']

            # Display results
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                st.metric("예측된 ΔE00", f"{delta_e:.3f}", delta=None)

                st.write("**예측 Lab 값:**")
                st.write(f"L*: {predicted_lab[0]:.2f}")
                st.write(f"a*: {predicted_lab[1]:.2f}")
                st.write(f"b*: {predicted_lab[2]:.2f}")

                st.write("")
                st.write("**Target Lab 값:**")
                st.write(f"L*: {target_lab[0, 0]:.2f}")
                st.write(f"a*: {target_lab[0, 1]:.2f}")
                st.write(f"b*: {target_lab[0, 2]:.2f}")

            with col_result2:
                st.write("**색상 비교:**")
                fig_compare = show_color_patches(target_lab.flatten(), predicted_lab.reshape(-1))
                st.pyplot(fig_compare)

            # Show recipe used
            with st.expander("📊 사용된 레시피 상세"):
                recipe_series = pd.Series(recipe_array.flatten(), index=CONFIG['recipe_cols'])
                recipe_nonzero = recipe_series[recipe_series > 0].sort_values(ascending=False)

                if not recipe_nonzero.empty:
                    recipe_display_df = pd.DataFrame({
                        'PIGMENT': recipe_nonzero.index,
                        '함량 (g/K)': recipe_nonzero.values
                    }).reset_index(drop=True)

                    st.dataframe(
                        recipe_display_df.style.format({'함량 (g/K)': '{:.4f}'}),
                        hide_index=True,
                        use_container_width=True
                    )

# 모델 로드 실패 시
else:
    st.error("‼️ 모델 로딩 실패.")
    st.code("""
    [필수 파일 목록]
    1. app_v2.py (지금 이 파일)
    2. recipe_model_optimized_weight_0.1.pth (⭐️ NEW)
    3. name_encoder.pkl
    4. xgb_surrogate_3.pkl (레시피 예측용)
    5. xgb_surrogate_3.pkl (시뮬레이터용)
    6. swatch_recipe_merged_1120.csv (참조 데이터셋)
    7. requirements.txt
    """)
