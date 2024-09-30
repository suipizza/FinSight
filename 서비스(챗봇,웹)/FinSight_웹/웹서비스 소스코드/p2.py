import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import yfinance as yf
from collections import OrderedDict

# 페이지 설정
st.set_page_config(layout="wide", page_title="Fin Sight 주가 예측 서비스")

# 색상 변수 정의
MAIN_COLOR = "#6a4abd"  # 도지블루

# 전체 앱에 대한 폰트 크기 증가 및 굵게 설정, 색상 통일
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-size: 28px;
        font-weight: bold;
    }}
    .stButton > button {{
        font-size: 28px;
        font-weight: bold;
    }}
    .stSelectbox > div > div > select {{
        font-size: 28px;
        font-weight: bold;
    }}
    .section-title {{
        font-size: 40px;
        font-weight: bold;
        color: {MAIN_COLOR};
        padding: 20px 0;
        border-bottom: 2px solid {MAIN_COLOR};
        margin-bottom: 20px;
    }}
    .subsection-title {{
        font-size: 34px;
        font-weight: bold;
        color: {MAIN_COLOR};
        padding: 15px 0;
        margin-top: 30px;
    }}
</style>
""", unsafe_allow_html=True)

# 메뉴 옵션
menu_options = [
    {"icon": "house", "label": "홈"},
    {"icon": "graph-up", "label": "모델 소개"},
    {"icon": "database", "label": "데이터셋 소개"},
    {"icon": "lightning", "label": "예측"}
]

# 상단 메뉴 구성 (글자 크기 및 굵기 증가, 색상 통일)
selected = option_menu(
    menu_title=None,
    options=[option["label"] for option in menu_options],
    icons=[option["icon"] for option in menu_options],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f0f0"},
        "icon": {"color": "black", "font-size": "30px"}, 
        "nav-link": {
            "font-size": "28px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": f"{MAIN_COLOR}",
            "font-weight": "bold",
            "background-color": "#ffffff",
            "color": "black",
        },
        "nav-link-selected": {
            "background-color": f"{MAIN_COLOR}",
            "color": "white",
        },
    }
)

# 섹터 및 종목 데이터
sectors = OrderedDict([
    ("IT", ["Apple Inc. (AAPL)"]),
    ("필수 소비재", ["The Coca-Cola Co. (KO)"]),
    ("헬스케어", ["Johnson & Johnson (JNJ)"]),
    ("금융", ["JPMorgan Chase & Co. (JPM)"])
])

# 함수 정의
def home():
    st.title("Fin Sight 주가 예측 서비스")
    st.markdown("""
    # 딥러닝 기반 주식 가격 예측 플랫폼
    
    딥러닝 기술을 활용하여 주식 시장의 동향을 분석하고 예측합니다. 
    """)

    st.markdown('<div class="section-title">최근 시장 동향</div>', unsafe_allow_html=True)
    market_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'S&P 500': np.cumsum(np.random.randn(365) * 0.1) + 100,
        'NASDAQ': np.cumsum(np.random.randn(365) * 0.2) + 100,
        'DOW JONES': np.cumsum(np.random.randn(365) * 0.15) + 100
    })
    fig = px.line(market_data, x='Date', y=['S&P 500', 'NASDAQ', 'DOW JONES'], title='주요 지수 동향')
    fig.update_layout(height=600, font=dict(size=24, color="black", family="Arial Black"))
    fig.update_traces(line=dict(width=4))
    fig.update_xaxes(title_font=dict(size=28), tickfont=dict(size=24))
    fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=24))
    fig.update_layout(legend=dict(font=dict(size=24)))
    st.plotly_chart(fig, use_container_width=True)

def model_explanation():
    st.title("GRU 모델 설명")
    
    st.markdown('<div class="section-title">GRU(Gated Recurrent Unit)</div>', unsafe_allow_html=True)
    st.markdown("""
    GRU는 순환 신경망(RNN)의 변형으로, 
    시계열 데이터 처리에 탁월한 성능을 보입니다.
    특히 장기 의존성 문제를 효과적으로 해결하여 
    복잡한 시퀀스 데이터 분석에 적합합니다.
    """)
    
    st.markdown('<div class="subsection-title">GRU의 구조</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        GRU 셀의 주요 구성 요소:
        1. **Update Gate**: 새로운 정보 반영 정도 결정
        2. **Reset Gate**: 과거 정보 무시 정도 결정
        3. **Hidden State**: 현재 시점의 정보 저장
        """)
    
    with col2:
        # GRU 셀 구조 다이어그램 (SVG) - 크기 및 색상 조정
        gru_cell_svg = """
        <svg width="100%" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="95%" height="280" fill="#E6F3FF" opacity="0.7" stroke="#0066CC" stroke-width="3"/>
            <circle cx="50%" cy="50%" r="100" fill="#FFE6E6" stroke="#CC0000" stroke-width="3"/>
            <text x="50%" y="50%" text-anchor="middle" fill="#CC0000" font-size="28" font-weight="bold">Hidden State</text>
            <rect x="20" y="20" width="150" height="60" fill="#E6FFE6" stroke="#006600" stroke-width="3"/>
            <text x="95" y="55" text-anchor="middle" fill="#006600" font-size="24" font-weight="bold">Update Gate</text>
            <rect x="20" y="220" width="150" height="60" fill="#FFE6F0" stroke="#CC0066" stroke-width="3"/>
            <text x="95" y="255" text-anchor="middle" fill="#CC0066" font-size="24" font-weight="bold">Reset Gate</text>
            <line x1="170" y1="50" x2="230" y2="100" stroke="black" stroke-width="3" marker-end="url(#arrowhead)"/>
            <line x1="170" y1="250" x2="230" y2="200" stroke="black" stroke-width="3" marker-end="url(#arrowhead)"/>
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
        </svg>
        """
        st.components.v1.html(gru_cell_svg, height=300)
    
    st.markdown('<div class="subsection-title">모델 성능 비교</div>', unsafe_allow_html=True)
    st.markdown("""
    다양한 시계열 예측 모델의 Test RMSE를 비교해 보겠습니다. 
    RMSE(Root Mean Square Error)는 예측값과 실제값의 차이를 나타내는 지표로, 
    낮을수록 예측 정확도가 높음을 의미합니다.
    """)
    
    comparison_data = pd.DataFrame({
        'Model': ['GRU', 'LSTM', 'Transformer', 'XGB'],
        'Test RMSE': [0.0144, 0.022, 0.0572, 0.0674]
    })
    
    fig = px.bar(comparison_data, x='Model', y='Test RMSE', title='D1 Dataset: Test RMSE Comparison')
    fig.update_layout(
        height=500,
        xaxis_title='Model', 
        yaxis_title='Test RMSE',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor='lightgrey',
        font=dict(size=24, color="black", family="Arial Black"),
        title=dict(font=dict(size=30))
    )
    fig.update_traces(marker_color=['#FF3333', '#3333FF', '#33FF33', '#FF9933'], marker_line_width=3, opacity=0.8)
    fig.update_xaxes(title_font=dict(size=28), tickfont=dict(size=24))
    fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=24))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    위 그래프에서 볼 수 있듯이, GRU 모델은 다른 모델들과 비교했을 때 가장 우수한 성능을 보입니다.

    - GRU의 Test RMSE는 0.0144로, 두 번째로 좋은 성능을 보인 LSTM(0.022)보다도 약 34.5% 낮은 오차를 보여줍니다.
    - 이는 GRU가 주가 예측 태스크에서 더 정확한 예측을 할 수 있음을 의미합니다.
    - Transformer(0.0572)와 XGB(0.0674) 모델은 GRU에 비해 현저히 높은 RMSE를 보이고 있습니다.
    - 이는 이 특정 데이터셋과 태스크에 대해 GRU가 더 적합한 모델임을 시사합니다.
    """)

    st.markdown('<div class="subsection-title">GRU의 주가 예측 적용</div>', unsafe_allow_html=True)
    st.markdown("""
    GRU 모델이 주가 예측에 특히 효과적인 이유:
    """)
    
    reasons = [
        "**장기 의존성 포착**: 과거의 중요한 정보를 오랫동안 기억할 수 있어, 주가의 장기적 트렌드를 잘 파악합니다.",
        "**노이즈 필터링**: Update Gate와 Reset Gate를 통해 불필요한 정보를 걸러내어, 주가의 일시적 변동에 덜 민감합니다.",
        "**비선형성 모델링**: 복잡한 주가 패턴을 효과적으로 학습할 수 있어, 시장의 다양한 상황에 대응 가능합니다.",
        "**계산 효율성**: LSTM보다 단순한 구조로 비슷한 성능을 내기 때문에, 실시간 예측이나 빠른 모델 업데이트에 유리합니다."
    ]
    
    for reason in reasons:
        st.markdown(f"- {reason}")

    st.markdown("""
    이러한 특성들이 결합되어 GRU가 다른 모델들보다 더 낮은 RMSE를 달성할 수 있었으며, 
    이는 주가 예측에 있어 GRU의 우수성을 입증합니다.
    """)

def dataset_explanation():
    st.title("데이터셋 설명")
    st.markdown("""
    주가 예측을 위해 세 가지 주요 데이터셋을 활용합니다. 각 데이터셋은 서로 다른 측면의 정보를 제공하여 
    모델의 예측 성능을 향상시킵니다.
    """)
    
    # 실제 데이터 가져오기
    stock = yf.Ticker("AAPL")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 최근 10년
    data = stock.history(start=start_date, end=end_date)
    
    st.markdown('<div class="section-title">D1: 종가 + 기술지표</div>', unsafe_allow_html=True)
    st.markdown("""
    D1 데이터셋은 주식의 종가와 다양한 기술적 지표를 포함합니다. 이 데이터셋은 주로 주가의 과거 패턴과 
    추세를 분석하는 데 사용됩니다.
    """)
    
    st.markdown('<div class="subsection-title">주요 특성</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **종가**")
        st.markdown("- **거래량**")
    with col2:
        st.markdown("- **RSI**")
        st.markdown("- **MACD**")
        st.markdown("- **볼린저 밴드**")
    
    st.markdown('<div class="subsection-title">D1 데이터셋 시각화</div>', unsafe_allow_html=True)
    
    # RSI 계산
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 계산
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=("종가", "거래량", "RSI", "MACD"))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="종가", line=dict(color='#FF4136', width=3)), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="거래량", marker_color='#0074D9'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color='#2ECC40', width=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD", line=dict(color='#FF851B', width=3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], name="Signal Line", line=dict(color='#B10DC9', width=3)), row=4, col=1)
    
    fig.update_layout(height=1200, font=dict(size=28, color="black", family="Arial Black"))  # 그래프 높이 증가 및 폰트 크기 조정
    fig.update_xaxes(title_font=dict(size=32), tickfont=dict(size=28))
    fig.update_yaxes(title_font=dict(size=32), tickfont=dict(size=28))
    
    # 각 서브플롯의 제목 크기 조정
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=36, color="black", family="Arial Black")
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="subsection-title">D2 데이터셋 시각화</div>', unsafe_allow_html=True)
    
    # 데이터 가져오기 및 처리 함수
    def get_data(ticker, start_date, end_date):
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']
            return data
        except Exception as e:
            st.error(f"데이터를 가져오는 중 오류가 발생했습니다 ({ticker}): {str(e)}")
            return pd.Series()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 최근 10년
    
    # 금리 데이터 (10년물 국채 수익률)
    treasury_10y = get_data("^TNX", start_date, end_date)
    
    # 달러 인덱스 데이터
    dollar_index = get_data("DX-Y.NYB", start_date, end_date)
    
    # 2년물 국채 수익률 데이터
    treasury_2y = get_data("^IRX", start_date, end_date)  # ^TWO 대신 ^IRX 사용

    # 장단기 금리차 계산
    if not treasury_10y.empty and not treasury_2y.empty:
        common_dates = treasury_10y.index.intersection(treasury_2y.index)
        yield_spread = treasury_10y[common_dates] - treasury_2y[common_dates]
    else:
        yield_spread = pd.Series()

    # 그래프 생성
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("금리", "달러 인덱스", "장단기 금리차"))
    
    if not treasury_10y.empty:
        fig.add_trace(go.Scatter(x=treasury_10y.index, y=treasury_10y, name="금리", line=dict(color='#FF4136', width=3)), row=1, col=1)
    else:
        st.warning("금리 데이터를 가져오지 못했습니다.")

    if not dollar_index.empty:
        fig.add_trace(go.Scatter(x=dollar_index.index, y=dollar_index, name="달러 인덱스", line=dict(color='#0074D9', width=3)), row=2, col=1)
    else:
        st.warning("달러 인덱스 데이터를 가져오지 못했습니다.")

    if not yield_spread.empty:
        fig.add_trace(go.Scatter(x=yield_spread.index, y=yield_spread, name="장단기 금리차", line=dict(color='#2ECC40', width=3)), row=3, col=1)
    else:
        st.warning("장단기 금리차 데이터를 계산하지 못했습니다.")
    
    fig.update_layout(height=1200, showlegend=False, font=dict(size=28, color="black", family="Arial Black"))
    fig.update_yaxes(title_text="금리 (%)", row=1, col=1, title_font=dict(size=32), tickfont=dict(size=28))
    fig.update_yaxes(title_text="달러 인덱스", row=2, col=1, title_font=dict(size=32), tickfont=dict(size=28))
    fig.update_yaxes(title_text="금리차 (%p)", row=3, col=1, title_font=dict(size=32), tickfont=dict(size=28))
    fig.update_xaxes(title_text="날짜", row=3, col=1, title_font=dict(size=32), tickfont=dict(size=28))
    
    # 각 서브플롯의 제목 크기 조정
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=36, color="black", family="Arial Black")
    
    st.plotly_chart(fig, use_container_width=True)

    
    st.markdown('<div class="section-title">D3: 종가 + 기술지표 + 외부요인</div>', unsafe_allow_html=True)
    st.markdown("""
    D3 데이터셋은 D1과 D2의 모든 특성을 결합한 가장 포괄적인 데이터셋입니다. 이 데이터셋은 기술적 분석과 
    기본적 분석을 모두 고려하여 가장 정확한 예측을 제공할 수 있습니다.
    """)
    
    st.markdown('<div class="subsection-title">주요 특성</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- **종가**")
        st.markdown("- **거래량**")
    with col2:
        st.markdown("- **RSI**")
        st.markdown("- **MACD**")
        st.markdown("- **볼린저 밴드**")
    with col3:
        st.markdown("- **장단기 금리차**")
        st.markdown("- **달러 인덱스**")
        st.markdown("- **금리**")
    
    st.markdown('<div class="section-title">데이터 분할 (Train/Validation/Test)</div>', unsafe_allow_html=True)
    
    total_samples = len(data)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], name='Train', mode='lines', line=dict(color='#FF4136', width=3)))
    fig.add_trace(go.Scatter(x=val_data.index, y=val_data['Close'], name='Validation', mode='lines', line=dict(color='#0074D9', width=3)))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], name='Test', mode='lines', line=dict(color='#2ECC40', width=3)))
    
    fig.update_layout(
        height=600,
        title=dict(text='주가 데이터 분할 (6:2:2)', font=dict(size=36, color="black", family="Arial Black")),
        xaxis_title='날짜',
        yaxis_title='종가',
        legend_title='데이터셋',
        font=dict(size=28, color="black", family="Arial Black")
    )
    fig.update_xaxes(title_font=dict(size=32), tickfont=dict(size=28))
    fig.update_yaxes(title_font=dict(size=32), tickfont=dict(size=28))
    st.plotly_chart(fig, use_container_width=True)

def prediction():
    st.title("주가 예측")
    
    st.markdown('<div class="subsection-title">섹터 선택</div>', unsafe_allow_html=True)
    sector = st.selectbox("", list(sectors.keys()), key='sector_select')
    
    st.markdown('<div class="subsection-title">종목 선택</div>', unsafe_allow_html=True)
    stock = st.selectbox("", sectors[sector], key='stock_select')

    if stock:
        st.markdown(f'<div class="section-title">{stock} 주가 예측</div>', unsafe_allow_html=True)
        
        ticker = stock.split('(')[1].split(')')[0]
        
        # 최근 30일간의 주가 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        current_price = stock_data['Close'].iloc[-1]
        
        predictions_data = {
            "AAPL": [214.18, 209.28, 207.48, 206.65, 206.26],
            "KO": [68.63, 66.72, 65.40, 64.65, 64.24],
            "JNJ": [160.30, 158.46, 156.09, 154.50, 153.50],
            "JPM": [211.91, 212.11, 211.99, 211.93, 211.92]
        }
        
        prediction_dates = ["월", "화", "수", "목", "금"]
        predictions = predictions_data[ticker]
        
        # 해당 주식의 월요일 주가 가져오기
        monday_prices = {
            "AAPL": 225.89,
            "KO": 68.98,
            "JNJ": 159.63,
            "JPM": 215.45
        }
        monday_price = monday_prices[ticker]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-title">현재 주가 정보</div>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='font-size: 32px; font-weight: bold;'>현재 가격: ${current_price:.2f}</h3>", unsafe_allow_html=True)
            
            # 최근 30일 주가 그래프 (크기 및 색상 조정)
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='실제 주가',
                                            line=dict(color='#FF4136', width=3)))
            fig_recent.update_layout(
                height=500,
                title=dict(text="최근 30일 주가 추이", font=dict(size=36, color="black", family="Arial Black")),
                xaxis_title="날짜",
                yaxis_title="주가 ($)",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white", font_size=24),
                template="plotly_white",
                font=dict(size=28, color="black", family="Arial Black")
            )
            fig_recent.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])],  # 주말 제외
                showgrid=True, gridwidth=1, gridcolor='lightgrey',
                title_font=dict(size=32), tickfont=dict(size=28)
            )
            fig_recent.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                                    title_font=dict(size=32), tickfont=dict(size=28))
            st.plotly_chart(fig_recent, use_container_width=True)
        
        with col2:
            st.markdown('<div class="subsection-title">주가 예측 결과</div>', unsafe_allow_html=True)
            next_day_prediction = predictions[0]
            delta = next_day_prediction - current_price
            st.markdown(f"<h3 style='font-size: 32px; font-weight: bold;'>다음 거래일 예상 가격: ${next_day_prediction:.2f} (변화: {delta:.2f})</h3>", unsafe_allow_html=True)
            
            if delta > 0:
                st.markdown(f"<h3 style='font-size: 32px; font-weight: bold; color: green; background-color: #e6ffe6; padding: 15px;'>주가가 상승할 것으로 예상됩니다.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 32px; font-weight: bold; color: red; background-color: #ffe6e6; padding: 15px;'>주가가 하락할 것으로 예상됩니다.</h3>", unsafe_allow_html=True)
        
        # 주가 예측 그래프 (크기 및 색상 조정)
        st.markdown('<div class="subsection-title">주가 예측 그래프</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, name='예측 주가',
                                 line=dict(color='#FF4136', width=3)))
        fig.add_trace(go.Scatter(x=prediction_dates, y=[monday_price] + [None]*4, name='실제 주가 (월요일)',
                                 mode='markers', marker=dict(color='#0074D9', size=16)))
        fig.update_layout(
            height=600,
            title=dict(text="주가 예측 vs 실제 주가 (월-금)", font=dict(size=36, color="black", family="Arial Black")),
            xaxis_title="요일",
            yaxis_title="주가 ($)",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font_size=28),
            template="plotly_white",
            font=dict(size=28, color="black", family="Arial Black")
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         title_font=dict(size=32), tickfont=dict(size=28))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         title_font=dict(size=32), tickfont=dict(size=28))
        st.plotly_chart(fig, use_container_width=True)

# 메인 앱 로직
if selected == "홈":
    home()
elif selected == "모델 소개":
    model_explanation()
elif selected == "데이터셋 소개":
    dataset_explanation()
elif selected == "예측":
    prediction()