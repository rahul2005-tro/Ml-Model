import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Solar Panel Maintenance",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)



DATA_REDUCTION_FACTOR = 4 
SIMPLIFIED_PLOTS = True    
MAX_POINTS_PER_PLOT = 1000



@st.cache_data
def generate_lightweight_data():
    """
    Generate lighter synthetic solar panel data optimized for Raspberry Pi.
    Uses 6-hour intervals instead of hourly to reduce memory usage.
    """
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31, 23, 59, 59)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='6H')  
    
    panel_ids = ['panel_1', 'panel_2', 'panel_3']
    all_data = []
    
    progress_bar = st.progress(0)
    total_operations = len(panel_ids) * len(timestamps)
    current_op = 0
    
    for panel_id in panel_ids:
        for timestamp in timestamps:
            current_op += 1
            if current_op % 100 == 0:  
                progress_bar.progress(current_op / total_operations)
            
            hour = timestamp.hour
            day_of_year = timestamp.dayofyear
            
            if 6 <= hour <= 18:
                daily_ghi = 800 * np.sin(np.pi * (hour - 6) / 12)
                seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                ghi = daily_ghi * seasonal_factor + np.random.normal(0, 30)  
                ghi = max(0, ghi)
            else:
                ghi = max(0, np.random.normal(0, 5))
            
            daily_temp = 15 + 10 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 10
            seasonal_temp = 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temp_c = daily_temp + seasonal_temp + np.random.normal(0, 1.5)
            
            base_power = ghi * 0.005
            temp_effect = -0.002 * max(0, temp_c - 25)
            power_output = base_power * (1 + temp_effect) + np.random.normal(0, 0.05)
            power_output = max(0, power_output)
            
            all_data.append({
                'timestamp': timestamp,
                'panel_id': panel_id,
                'ghi': round(ghi, 2),
                'temp_c': round(temp_c, 2),
                'power_output_kw': round(power_output, 3)
            })
    
    progress_bar.progress(1.0)
    df = pd.DataFrame(all_data)
    
    anomaly_periods = [
        (datetime(2023, 3, 15), datetime(2023, 3, 16)),
        (datetime(2023, 7, 10), datetime(2023, 7, 11)),
    ]
    
    for start, end in anomaly_periods:
        mask = (df['panel_id'] == 'panel_2') & (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df.loc[mask, 'power_output_kw'] *= 0.75  
    
    return df


def create_simple_time_series(df, y_col, title, max_points=MAX_POINTS_PER_PLOT):
    """Create simplified time series plot optimized for Pi"""
    if len(df) > max_points:
        df_sample = df.sample(n=max_points).sort_values('timestamp')
    else:
        df_sample = df
    
    fig = px.line(df_sample, x='timestamp', y=y_col, color='panel_id', title=title)
    fig.update_layout(height=400, showlegend=True)
    fig.update_traces(line_width=1)
    return fig

def create_simple_histogram(df, col, title, bins=30):
    """Create simplified histogram"""
    fig = px.histogram(df, x=col, title=title, nbins=bins)
    fig.update_layout(height=300)
    return fig

def create_simple_correlation_heatmap(df):
    """Create simplified correlation heatmap"""
    numeric_cols = ['ghi', 'temp_c', 'power_output_kw', 'hour', 'month']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Feature Correlations",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(height=400)
    return fig



def engineer_basic_features(df):
    """Create essential time-based features"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    return df



def train_lightweight_model(df):
    """Train Random Forest model (better Pi compatibility than LightGBM)"""
    feature_cols = ['ghi', 'temp_c', 'hour', 'month', 'is_weekend']
    X = df[feature_cols]
    y = df['power_output_kw']
    
    split_date = datetime(2023, 10, 1)  
    train_mask = df['timestamp'] < split_date
    
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    model = RandomForestRegressor(
        n_estimators=50, 
        max_depth=10,
        random_state=42,
        n_jobs=1  
    )
    
    with st.spinner("Training model on Raspberry Pi..."):
        model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    return model, y_pred_all, mae, rmse

def train_lightweight_anomaly_detection(df, predictions):
    """Train simplified Isolation Forest"""
    df = df.copy()
    df['predicted_power'] = predictions
    df['deviation'] = df['power_output_kw'] - df['predicted_power']
    
    iso_forest = IsolationForest(
        contamination=0.02, 
        n_estimators=50,     
        random_state=42
    )
    
    with st.spinner("Training anomaly detector..."):
        anomaly_flags = iso_forest.fit_predict(df[['deviation']])
    
    df['anomaly_flag'] = anomaly_flags
    return df, iso_forest



def main():
    st.title("Solar Panel Monitor (Pi Edition)")
    st.markdown("*Optimized for Raspberry Pi*")
    st.markdown("---")
    
    with st.expander(" System Information"):
        st.info("""
        **Pi Optimizations Active:**
        - 6-hour data intervals (vs hourly)
        - Random Forest (vs LightGBM) 
        - Simplified visualizations
        - Reduced memory usage
        """)
    
    st.subheader(" Loading Data...")
    
    df = generate_lightweight_data()
    st.success(f"Loaded {len(df):,} data points")
    
    df = engineer_basic_features(df)
    
    st.subheader("Training Models...")
    model, predictions, mae, rmse = train_lightweight_model(df)
    df_with_anomalies, iso_forest = train_lightweight_anomaly_detection(df, predictions)
    st.success("Models trained successfully!")
    
    st.sidebar.header(" Controls")
    selected_panels = st.sidebar.multiselect(
        "Select Panels",
        options=df['panel_id'].unique(),
        default=df['panel_id'].unique()
    )
    
    filtered_df = df_with_anomalies[df_with_anomalies['panel_id'].isin(selected_panels)]
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Predictions", "🚨 Anomalies"])

    with tab1:
        st.header(" Data Overview")
        
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df[['ghi', 'temp_c', 'power_output_kw']].describe())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_power = create_simple_time_series(filtered_df, 'power_output_kw', "Power Output")
            st.plotly_chart(fig_power, use_container_width=True)
            
            fig_power_hist = create_simple_histogram(filtered_df, 'power_output_kw', "Power Distribution")
            st.plotly_chart(fig_power_hist, use_container_width=True)
        
        with col2:
            fig_ghi = create_simple_time_series(filtered_df, 'ghi', "Solar Irradiance")
            st.plotly_chart(fig_ghi, use_container_width=True)
            
            fig_corr = create_simple_correlation_heatmap(filtered_df)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        if st.checkbox("Show Data Sample"):
            st.dataframe(filtered_df.head(50))
    

    with tab2:
        st.header(" Performance Prediction")
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Random Forest")
        with col2:
            st.metric("MAE", f"{mae:.3f} kW")
        with col3:
            st.metric("RMSE", f"{rmse:.3f} kW")
        
        st.subheader("Actual vs Predicted")
        
        plot_data = filtered_df.sample(n=min(500, len(filtered_df))).sort_values('timestamp')
        
        fig = go.Figure()
        
        for panel in selected_panels:
            panel_data = plot_data[plot_data['panel_id'] == panel]
            
            fig.add_trace(go.Scatter(
                x=panel_data['timestamp'],
                y=panel_data['power_output_kw'],
                mode='markers',
                name=f'{panel} - Actual',
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=panel_data['timestamp'],
                y=panel_data['predicted_power'],
                mode='lines',
                name=f'{panel} - Predicted',
                line=dict(dash='dot')
            ))
        
        fig.update_layout(
            title="Power Output: Actual vs Predicted (Sample)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    

    with tab3:
        st.header("Anomaly Detection")
        
        anomalies = filtered_df[filtered_df['anomaly_flag'] == -1]
        st.metric("Anomalies Detected", len(anomalies))
        
        if len(anomalies) > 0:
            fig = go.Figure()
            
            for panel in selected_panels:
                panel_data = filtered_df[filtered_df['panel_id'] == panel]
                normal_data = panel_data[panel_data['anomaly_flag'] == 1]
                anomaly_data = panel_data[panel_data['anomaly_flag'] == -1]
                
                if len(normal_data) > 300:
                    normal_data = normal_data.sample(n=300).sort_values('timestamp')
                
                fig.add_trace(go.Scatter(
                    x=normal_data['timestamp'],
                    y=normal_data['power_output_kw'],
                    mode='lines',
                    name=f'{panel} - Normal',
                    opacity=0.7
                ))
                
                if len(anomaly_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['power_output_kw'],
                        mode='markers',
                        name=f'{panel} - Anomaly',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
            
            fig.update_layout(
                title="Power Output with Anomalies (Red X marks)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Anomaly Details")
            anomaly_display = anomalies[['timestamp', 'panel_id', 'power_output_kw', 'predicted_power', 'deviation']]
            st.dataframe(anomaly_display)
            
        else:
            st.info("No anomalies detected.")



def show_pi_performance_tips():
    """Display performance tips for Raspberry Pi users"""
    with st.sidebar.expander("Pi Performance Tips"):
        st.markdown("""
        **To improve performance:**
        - Close other applications
        - Use ethernet vs WiFi
        - Ensure adequate cooling
        - Consider Pi 4 (4GB+ RAM)
        - Run: `sudo apt update && sudo apt upgrade`
        """)



if __name__ == "__main__":
    show_pi_performance_tips()
    main()