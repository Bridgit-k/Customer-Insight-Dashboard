"""
Multi-Branch Customer Loyalty Dashboard - Enterprise Edition
Features: Multi-branch/department, KES currency, Benchmark ranking system, Visit pattern analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Customer Loyalty Dashboard",
    page_icon="🏢",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(uploaded_file):
    """Load data from CSV or Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            df = pd.read_excel(uploaded_file)
        
        if df.empty:
            st.error("The file is empty!")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def safe_to_datetime(series):
    """Safely convert series to datetime"""
    try:
        converted = pd.to_datetime(series, errors='coerce')
        if converted.isna().all() and pd.api.types.is_numeric_dtype(series):
            try:
                converted = pd.to_datetime('1899-12-30') + pd.to_timedelta(series, 'D')
            except:
                pass
        return converted
    except:
        return series

def safe_to_numeric(series):
    """Safely convert series to numeric"""
    try:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        
        if series.dtype == 'object':
            series = series.astype(str)
            for symbol in ['KES', 'Ksh', 'KSh', '$', '£', '€']:
                series = series.str.replace(symbol, '', regex=False)
            series = series.str.replace(',', '', regex=False)
            series = series.str.strip()
        
        return pd.to_numeric(series, errors='coerce')
    except:
        return pd.to_numeric(series, errors='coerce')

def prepare_data(df, customer_col, date_col, amount_col, branch_col=None, dept_col=None):
    """Clean and prepare data"""
    try:
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        df_clean[date_col] = safe_to_datetime(df_clean[date_col])
        df_clean['hour'] = pd.to_datetime(df_clean[date_col]).dt.hour
        df_clean['day_of_week'] = pd.to_datetime(df_clean[date_col]).dt.day_name()
        df_clean['is_weekend'] = pd.to_datetime(df_clean[date_col]).dt.dayofweek >= 5
        df_clean['date_only'] = pd.to_datetime(df_clean[date_col]).dt.date
        df_clean['year_month'] = pd.to_datetime(df_clean[date_col]).dt.to_period('M')
        
        df_clean[amount_col] = safe_to_numeric(df_clean[amount_col])
        
        if branch_col and branch_col in df_clean.columns:
            df_clean[branch_col] = df_clean[branch_col].fillna('No Branch Specified')
        
        if dept_col and dept_col in df_clean.columns:
            df_clean[dept_col] = df_clean[dept_col].fillna('No Department Specified')
        
        df_clean = df_clean.dropna(subset=[customer_col, date_col, amount_col])
        df_clean = df_clean[df_clean[amount_col] > 0]
        df_clean = df_clean.drop_duplicates()
        
        if len(df_clean) == 0:
            st.error("No valid data after cleaning!")
            return None
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            st.info(f"🧹 Cleaned: {rows_removed} invalid rows removed ({initial_rows} → {len(df_clean)} rows)")
        
        return df_clean
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def calculate_loyalty_scores_with_benchmark(df, customer_col, date_col, amount_col, 
                                           ideal_visits_per_month=6, ideal_monthly_spend=100000,
                                           freq_weight=0.6, spend_weight=0.4):
    """
    Calculate loyalty scores using an ideal benchmark customer
    
    Ideal Customer Benchmark:
    - Visits per month: ideal_visits_per_month (default 6)
    - Monthly spending: ideal_monthly_spend (default KES 100,000)
    """
    try:
        # Calculate total months in dataset
        min_date = pd.to_datetime(df[date_col].min())
        max_date = pd.to_datetime(df[date_col].max())
        total_months = max((max_date.year - min_date.year) * 12 + max_date.month - min_date.month, 1)
        
        # Customer metrics
        customer_metrics = df.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            date_col: ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = [customer_col, 'total_spent', 'avg_spent', 'visit_count', 'first_visit', 'last_visit']
        
        # Calculate monthly metrics
        customer_metrics['visits_per_month'] = customer_metrics['visit_count'] / total_months
        customer_metrics['spending_per_month'] = customer_metrics['total_spent'] / total_months
        
        # BENCHMARK SCORING SYSTEM
        # Compare each customer to the ideal customer
        
        # Frequency Score: Compare visits per month to ideal (6+)
        # Score = (actual / ideal) * 100, capped at 100
        customer_metrics['frequency_score'] = (customer_metrics['visits_per_month'] / ideal_visits_per_month * 100).clip(upper=100)
        
        # Spending Score: Compare monthly spending to ideal (KES 100,000+)
        # Score = (actual / ideal) * 100, capped at 100
        customer_metrics['spending_score'] = (customer_metrics['spending_per_month'] / ideal_monthly_spend * 100).clip(upper=100)
        
        # Weighted Loyalty Score
        customer_metrics['loyalty_score'] = (
            customer_metrics['frequency_score'] * freq_weight +
            customer_metrics['spending_score'] * spend_weight
        )
        
        # Recency
        current_date = pd.to_datetime(df[date_col].max()) + timedelta(days=1)
        customer_metrics['days_since_last_visit'] = (current_date - pd.to_datetime(customer_metrics['last_visit'])).dt.days
        
        # TIER ASSIGNMENT (Based on loyalty score)
        def assign_tier(score):
            if score >= 80:
                return '🏆 VIP Champions'
            elif score >= 60:
                return '⭐ Loyal Customers'
            elif score >= 40:
                return '💎 Growing Customers'
            else:
                return '🌱 New/Casual Customers'
        
        customer_metrics['loyalty_tier'] = customer_metrics['loyalty_score'].apply(assign_tier)
        
        # LEVEL ASSIGNMENT (Based on rank percentiles)
        # Sort by loyalty score to get ranks
        customer_metrics = customer_metrics.sort_values('loyalty_score', ascending=False)
        customer_metrics['rank'] = range(1, len(customer_metrics) + 1)
        
        # Calculate percentile for each customer
        total_customers = len(customer_metrics)
        customer_metrics['percentile'] = (customer_metrics['rank'] / total_customers * 100)
        
        # Assign levels based on percentile
        def assign_level(percentile):
            if percentile <= 33.33:  # Top 33%
                return '🔥 Highest Level'
            elif percentile <= 66.67:  # Middle 33%
                return '⚡ Mid-Level'
            else:  # Bottom 33%
                return '🌟 Beginner Level'
        
        customer_metrics['level'] = customer_metrics['percentile'].apply(assign_level)
        
        # Calculate how close they are to ideal
        customer_metrics['frequency_vs_ideal'] = (customer_metrics['visits_per_month'] / ideal_visits_per_month * 100).round(1)
        customer_metrics['spending_vs_ideal'] = (customer_metrics['spending_per_month'] / ideal_monthly_spend * 100).round(1)
        
        return customer_metrics
    
    except Exception as e:
        st.error(f"Error calculating loyalty scores: {e}")
        return pd.DataFrame()

def analyze_shopping_patterns(df, customer_col):
    """Analyze weekday vs weekend shopping"""
    try:
        patterns = df.groupby(customer_col).agg({
            'is_weekend': lambda x: x.sum(),
            customer_col: 'count'
        }).reset_index()
        
        patterns.columns = [customer_col, 'weekend_visits', 'total_visits']
        patterns['weekday_visits'] = patterns['total_visits'] - patterns['weekend_visits']
        patterns['weekend_percentage'] = (patterns['weekend_visits'] / patterns['total_visits'] * 100).round(1)
        
        def classify_shopper(row):
            if row['weekend_percentage'] >= 70:
                return '🌴 Weekend Shopper'
            elif row['weekend_percentage'] >= 30:
                return '📊 Balanced Shopper'
            else:
                return '💼 Weekday Shopper'
        
        patterns['shopper_type'] = patterns.apply(classify_shopper, axis=1)
        return patterns
    except Exception as e:
        return pd.DataFrame()

def analyze_visit_patterns(df, customer_col, date_col):
    """Analyze when customers visit (days/times)"""
    try:
        visit_patterns = []
        
        for customer in df[customer_col].unique():
            customer_data = df[df[customer_col] == customer].copy()
            
            if len(customer_data) < 2:
                continue
            
            day_counts = customer_data['day_of_week'].value_counts()
            most_common_days = day_counts.head(2).index.tolist()
            
            hour_counts = customer_data['hour'].value_counts()
            most_common_hours = hour_counts.head(2).index.tolist()
            
            sorted_dates = customer_data[date_col].sort_values()
            date_diffs = sorted_dates.diff().dt.days.dropna()
            avg_days_between = date_diffs.mean() if len(date_diffs) > 0 else None
            
            visit_patterns.append({
                customer_col: customer,
                'total_visits': len(customer_data),
                'primary_day': most_common_days[0] if len(most_common_days) > 0 else 'Various',
                'secondary_day': most_common_days[1] if len(most_common_days) > 1 else '',
                'primary_hour': int(most_common_hours[0]) if len(most_common_hours) > 0 else None,
                'avg_days_between': round(avg_days_between, 1) if avg_days_between else None
            })
        
        patterns_df = pd.DataFrame(visit_patterns)
        
        def create_pattern_description(row):
            parts = []
            
            if row['avg_days_between'] and row['avg_days_between'] <= 7:
                parts.append("Weekly")
            elif row['avg_days_between'] and row['avg_days_between'] <= 14:
                parts.append("Bi-weekly")
            elif row['avg_days_between'] and row['avg_days_between'] <= 30:
                parts.append("Monthly")
            
            if row['secondary_day']:
                parts.append(f"{row['primary_day']}s & {row['secondary_day']}s")
            else:
                parts.append(f"{row['primary_day']}s")
            
            if row['primary_hour'] is not None:
                hour_12 = row['primary_hour'] % 12 or 12
                am_pm = 'AM' if row['primary_hour'] < 12 else 'PM'
                parts.append(f"~{hour_12}:00 {am_pm}")
            
            return " • ".join(parts)
        
        patterns_df['pattern'] = patterns_df.apply(create_pattern_description, axis=1)
        return patterns_df
    except Exception as e:
        return pd.DataFrame()

def format_currency(value):
    """Format as Kenyan Shillings"""
    return f"KES {value:,.2f}"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🏢 Customer Loyalty Dashboard - Enterprise Edition")
    st.markdown("""
    **Multi-Branch Analytics** • **Benchmark Ranking System** • **Kenyan Shillings** • **Visit Pattern Analysis**
    """)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    st.sidebar.header("📁 Upload Data")
    
    st.sidebar.subheader("🏢 Organization Structure")
    has_branches = st.sidebar.checkbox("My business has multiple branches")
    has_departments = st.sidebar.checkbox("My business has departments")
    
    if has_branches or has_departments:
        st.sidebar.subheader("📊 Data Organization")
        data_org = st.sidebar.radio(
            "How is your data organized?",
            ["📋 Consolidated file (with branch/dept columns)", 
             "📁 Separate files per branch/department"]
        )
        is_consolidated = "Consolidated" in data_org
    else:
        is_consolidated = True
    
    if is_consolidated:
        uploaded_file = st.sidebar.file_uploader("Upload data file", type=['csv', 'xlsx', 'xls'])
        uploaded_files = [uploaded_file] if uploaded_file else []
    else:
        uploaded_files = st.sidebar.file_uploader("Upload files", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)
    
    with st.sidebar.expander("📋 Data Format Guide"):
        st.markdown("""
        **Required:**
        - Customer ID/Name
        - Date & Time
        - Amount (KES)
        
        **Optional:**
        - Branch name
        - Department name
        
        **Example:**
        ```
        customer, date, amount, branch
        John, 2024-01-15 14:30, 5000, Nairobi
        ```
        """)
    
    # Benchmark settings
    st.sidebar.header("🎯 Ideal Customer Benchmark")
    ideal_visits = st.sidebar.number_input(
        "Ideal visits per month",
        min_value=1,
        max_value=30,
        value=6,
        help="The ideal customer visits this many times per month"
    )
    
    ideal_spend = st.sidebar.number_input(
        "Ideal monthly spending (KES)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="The ideal customer spends this much per month"
    )
    
    st.sidebar.info(f"""
    **Ideal Customer Profile:**
    - Visits: **{ideal_visits}+** times/month
    - Spends: **KES {ideal_spend:,}+**/month
    
    All customers are scored against this benchmark.
    """)
    
    # Rating weights
    st.sidebar.header("⚙️ Rating Weights")
    freq_weight = st.sidebar.slider("Frequency Importance", 0.0, 1.0, 0.6, 0.05)
    spend_weight = 1.0 - freq_weight
    
    st.sidebar.info(f"**Weights:** Frequency {freq_weight*100:.0f}% • Spending {spend_weight*100:.0f}%")
    
    # ========================================================================
    # NO DATA UPLOADED
    # ========================================================================
    
    if not uploaded_files or (is_consolidated and uploaded_files[0] is None):
        st.info("👆 Upload your data to begin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Single Business Sample")
            sample1 = pd.DataFrame({
                'customer': ['John', 'Jane', 'John', 'Bob', 'Jane', 'John'],
                'date': ['2024-01-15 14:30', '2024-01-20 10:00', '2024-01-25 15:45', 
                        '2024-02-01 09:00', '2024-02-10 16:30', '2024-02-15 11:00'],
                'amount': [5000, 3500, 4200, 7500, 6000, 5500]
            })
            st.dataframe(sample1, use_container_width=True)
            csv1 = sample1.to_csv(index=False)
            st.download_button("📥 Download", csv1, "sample_single.csv", "text/csv", use_container_width=True)
        
        with col2:
            st.subheader("📝 Multi-Branch Sample")
            sample2 = pd.DataFrame({
                'customer': ['John', 'Jane', 'Bob', 'Alice', 'John', 'Jane'],
                'date': ['2024-01-15 14:30', '2024-01-20 10:00', '2024-01-25 15:45',
                        '2024-02-01 09:00', '2024-02-10 16:30', '2024-02-15 11:00'],
                'amount': [5000, 3500, 4200, 7500, 6000, 8000],
                'branch': ['Nairobi CBD', 'Westlands', 'Nairobi CBD', 'Mombasa', 'Westlands', 'Nairobi CBD'],
                'department': ['Electronics', 'Clothing', 'Electronics', 'Furniture', 'Clothing', 'Electronics']
            })
            st.dataframe(sample2, use_container_width=True)
            csv2 = sample2.to_csv(index=False)
            st.download_button("📥 Download", csv2, "sample_multi.csv", "text/csv", use_container_width=True)
        
        return
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    all_data = []
    
    if is_consolidated:
        df = load_data(uploaded_files[0])
        if df is not None:
            all_data.append(df)
    else:
        for file in uploaded_files:
            df = load_data(file)
            if df is not None:
                df['_source_file'] = file.name
                all_data.append(df)
    
    if not all_data:
        st.error("Could not load data!")
        return
    
    df_combined = pd.concat(all_data, ignore_index=True)
    st.success(f"✅ Loaded {len(df_combined):,} rows")
    
    with st.expander("👀 Preview Data"):
        st.dataframe(df_combined.head(20), use_container_width=True)
    
    # ========================================================================
    # COLUMN MAPPING
    # ========================================================================
    
    st.sidebar.header("🔧 Map Columns")
    
    columns = df_combined.columns.tolist()
    
    default_customer = next((c for c in columns if any(w in c.lower() for w in ['customer', 'client', 'name', 'id'])), columns[0])
    default_date = next((c for c in columns if any(w in c.lower() for w in ['date', 'time'])), columns[1] if len(columns) > 1 else columns[0])
    default_amount = next((c for c in columns if any(w in c.lower() for w in ['amount', 'price', 'kes', 'ksh'])), columns[-1])
    
    customer_col = st.sidebar.selectbox("👤 Customer", columns, index=columns.index(default_customer))
    date_col = st.sidebar.selectbox("📅 Date/Time", columns, index=columns.index(default_date))
    amount_col = st.sidebar.selectbox("💰 Amount (KES)", columns, index=columns.index(default_amount))
    
    branch_col = None
    dept_col = None
    
    if has_branches:
        default_branch = next((c for c in columns if 'branch' in c.lower() or 'location' in c.lower()), None)
        branch_col = st.sidebar.selectbox("🏢 Branch", [None] + columns, 
                                         index=columns.index(default_branch) + 1 if default_branch else 0)
    
    if has_departments:
        default_dept = next((c for c in columns if 'dept' in c.lower() or 'department' in c.lower()), None)
        dept_col = st.sidebar.selectbox("🏪 Department", [None] + columns,
                                       index=columns.index(default_dept) + 1 if default_dept else 0)
    
    # ========================================================================
    # ANALYZE
    # ========================================================================
    
    if st.sidebar.button("🚀 Analyze Data", type="primary", use_container_width=True):
        
        df_clean = prepare_data(df_combined, customer_col, date_col, amount_col, branch_col, dept_col)
        
        if df_clean is None or len(df_clean) == 0:
            return
        
        st.success(f"✅ Ready! {len(df_clean):,} transactions, {df_clean[customer_col].nunique():,} customers")
        
        # ================================================================
        # FILTERS
        # ================================================================
        
        st.header("🎯 Filter Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if branch_col:
                branches = ['All Branches'] + sorted(df_clean[branch_col].unique().tolist())
                selected_branch = st.selectbox("🏢 Branch", branches)
            else:
                selected_branch = 'All Branches'
        
        with col2:
            if dept_col:
                depts = ['All Departments'] + sorted(df_clean[dept_col].unique().tolist())
                selected_dept = st.selectbox("🏪 Department", depts)
            else:
                selected_dept = 'All Departments'
        
        with col3:
            min_date = df_clean[date_col].min().date()
            max_date = df_clean[date_col].max().date()
            date_range = st.date_input("📅 Date Range", value=(min_date, max_date), 
                                      min_value=min_date, max_value=max_date)
        
        # Apply filters
        df_filtered = df_clean.copy()
        
        if selected_branch != 'All Branches' and branch_col:
            df_filtered = df_filtered[df_filtered[branch_col] == selected_branch]
        
        if selected_dept != 'All Departments' and dept_col:
            df_filtered = df_filtered[df_filtered[dept_col] == selected_dept]
        
        if len(date_range) == 2:
            df_filtered = df_filtered[
                (df_filtered['date_only'] >= date_range[0]) &
                (df_filtered['date_only'] <= date_range[1])
            ]
        
        if len(df_filtered) == 0:
            st.warning("No data matches filters")
            return
        
        st.divider()
        
        # ================================================================
        # CALCULATE METRICS
        # ================================================================
        
        with st.spinner("Analyzing..."):
            customer_metrics = calculate_loyalty_scores_with_benchmark(
                df_filtered, customer_col, date_col, amount_col,
                ideal_visits, ideal_spend, freq_weight, spend_weight
            )
            shopping_patterns = analyze_shopping_patterns(df_filtered, customer_col)
            visit_patterns = analyze_visit_patterns(df_filtered, customer_col, date_col)
        
        # ================================================================
        # KEY METRICS
        # ================================================================
        
        st.header("📊 Business Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_revenue = df_filtered[amount_col].sum()
        total_txns = len(df_filtered)
        total_customers = df_filtered[customer_col].nunique()
        avg_txn = df_filtered[amount_col].mean()
        avg_per_cust = total_revenue / total_customers if total_customers > 0 else 0
        
        col1.metric("Total Revenue", format_currency(total_revenue))
        col2.metric("Transactions", f"{total_txns:,}")
        col3.metric("Customers", f"{total_customers:,}")
        col4.metric("Avg Transaction", format_currency(avg_txn))
        col5.metric("Avg/Customer", format_currency(avg_per_cust))
        
        # Branch/Dept breakdown
        if (branch_col and selected_branch == 'All Branches') or (dept_col and selected_dept == 'All Departments'):
            st.divider()
            st.subheader("🏢 Performance Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if branch_col and selected_branch == 'All Branches':
                    branch_stats = df_filtered.groupby(branch_col)[amount_col].agg(['sum', 'count']).reset_index()
                    branch_stats.columns = ['Branch', 'Revenue', 'Transactions']
                    branch_stats = branch_stats.sort_values('Revenue', ascending=False).head(10)
                    
                    fig = px.bar(branch_stats, x='Branch', y='Revenue', title='Top 10 Branches by Revenue',
                                labels={'Revenue': 'Revenue (KES)'}, text='Transactions')
                    fig.update_traces(texttemplate='%{text} txns', textposition='outside')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if dept_col and selected_dept == 'All Departments':
                    dept_stats = df_filtered.groupby(dept_col)[amount_col].agg(['sum', 'count']).reset_index()
                    dept_stats.columns = ['Department', 'Revenue', 'Transactions']
                    dept_stats = dept_stats.sort_values('Revenue', ascending=False).head(10)
                    
                    fig = px.bar(dept_stats, x='Department', y='Revenue', title='Top 10 Departments by Revenue',
                                labels={'Revenue': 'Revenue (KES)'}, text='Transactions')
                    fig.update_traces(texttemplate='%{text} txns', textposition='outside')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ================================================================
        # TOP CUSTOMERS WITH BENCHMARK RANKING
        # ================================================================
        
        st.header("🏆 Customer Rankings (Benchmark-Based)")
        
        st.info(f"""
        **📊 Ranking System Explained:**
        
        Customers are scored against an **Ideal Customer Benchmark**:
        - **Ideal Visits:** {ideal_visits}+ per month
        - **Ideal Spending:** KES {ideal_spend:,}+ per month
        
        **Scoring:**
        - Frequency Score = (Your visits/month ÷ {ideal_visits}) × 100
        - Spending Score = (Your spending/month ÷ {ideal_spend:,}) × 100
        - Loyalty Score = (Frequency × {freq_weight*100:.0f}%) + (Spending × {spend_weight*100:.0f}%)
        
        **Example:** If a customer visits 3 times/month and spends KES 50,000/month:
        - Frequency Score: (3÷{ideal_visits})×100 = {3/ideal_visits*100:.1f}
        - Spending Score: (50,000÷{ideal_spend:,})×100 = {50000/ideal_spend*100:.1f}
        - Loyalty Score: {3/ideal_visits*100*freq_weight + 50000/ideal_spend*100*spend_weight:.1f}
        """)
        
        top_n = st.slider("Show top N customers", 10, 100, 20)
        top_customers = customer_metrics.head(top_n).copy()
        
        # Display table
        display_top = top_customers[[customer_col, 'visit_count', 'visits_per_month', 'total_spent', 
                                    'spending_per_month', 'frequency_score', 'spending_score', 
                                    'loyalty_score', 'rank', 'loyalty_tier', 'level']].copy()
        
        display_top['visits_per_month'] = display_top['visits_per_month'].apply(lambda x: f"{x:.1f}")
        display_top['total_spent'] = display_top['total_spent'].apply(lambda x: format_currency(x))
        display_top['spending_per_month'] = display_top['spending_per_month'].apply(lambda x: format_currency(x))
        display_top['frequency_score'] = display_top['frequency_score'].apply(lambda x: f"{x:.0f}")
        display_top['spending_score'] = display_top['spending_score'].apply(lambda x: f"{x:.0f}")
        display_top['loyalty_score'] = display_top['loyalty_score'].apply(lambda x: f"{x:.0f}")
        
        display_top.columns = ['Customer', 'Total Visits', 'Visits/Month', 'Total Spent', 
                               'Spending/Month', 'Freq Score', 'Spend Score', 
                               'Loyalty Score', 'Rank', 'Tier', 'Level']
        
        st.dataframe(display_top, use_container_width=True, hide_index=True, height=400)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(top_customers.head(15), x=customer_col, y='loyalty_score',
                        color='level', title='Top 15 Customers by Loyalty Score',
                        labels={customer_col: 'Customer', 'loyalty_score': 'Loyalty Score'},
                        color_discrete_map={
                            '🔥 Highest Level': '#FFD700',
                            '⚡ Mid-Level': '#C0C0C0',
                            '🌟 Beginner Level': '#CD7F32'
                        })
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tier and Level distribution
            tier_level_counts = customer_metrics.groupby(['loyalty_tier', 'level']).size().reset_index(name='count')
            
            fig = px.sunburst(tier_level_counts, path=['loyalty_tier', 'level'], values='count',
                            title='Customer Distribution: Tiers & Levels')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ================================================================
        # BENCHMARK COMPARISON VISUALIZATION
        # ================================================================
        
        st.header("📊 Performance vs Ideal Benchmark")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter: Visits per month vs Spending per month
            fig = go.Figure()
            
            for level in ['🔥 Highest Level', '⚡ Mid-Level', '🌟 Beginner Level']:
                level_data = customer_metrics[customer_metrics['level'] == level]
                
                fig.add_trace(go.Scatter(
                    x=level_data['visits_per_month'],
                    y=level_data['spending_per_month'],
                    mode='markers',
                    name=level,
                    text=level_data[customer_col],
                    marker=dict(size=10),
                    hovertemplate='<b>%{text}</b><br>Visits/month: %{x:.1f}<br>Spending/month: KES %{y:,.0f}<extra></extra>'
                ))
            
            # Add ideal benchmark line
            fig.add_shape(type='line', x0=ideal_visits, x1=ideal_visits, y0=0, y1=customer_metrics['spending_per_month'].max()*1.1,
                         line=dict(color='red', dash='dash', width=2), name='Ideal Visits')
            fig.add_shape(type='line', x0=0, x1=customer_metrics['visits_per_month'].max()*1.1, y0=ideal_spend, y1=ideal_spend,
                         line=dict(color='red', dash='dash', width=2), name='Ideal Spending')
            
            fig.add_annotation(x=ideal_visits, y=customer_metrics['spending_per_month'].max()*1.05,
                             text=f"Ideal: {ideal_visits} visits/month", showarrow=False, 
                             font=dict(color='red', size=10))
            fig.add_annotation(x=customer_metrics['visits_per_month'].max()*0.9, y=ideal_spend,
                             text=f"Ideal: KES {ideal_spend:,}/month", showarrow=False,
                             font=dict(color='red', size=10))
            
            fig.update_layout(
                title='Customer Performance vs Ideal Benchmark',
                xaxis_title='Visits per Month',
                yaxis_title='Spending per Month (KES)',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution of scores
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(x=customer_metrics['frequency_score'], name='Frequency Score',
                                      opacity=0.7, nbinsx=20))
            fig.add_trace(go.Histogram(x=customer_metrics['spending_score'], name='Spending Score',
                                      opacity=0.7, nbinsx=20))
            
            fig.update_layout(
                title='Distribution of Customer Scores',
                xaxis_title='Score (% of Ideal)',
                yaxis_title='Number of Customers',
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Stats summary
        st.subheader("📈 Benchmark Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        customers_meeting_freq = (customer_metrics['visits_per_month'] >= ideal_visits).sum()
        customers_meeting_spend = (customer_metrics['spending_per_month'] >= ideal_spend).sum()
        customers_meeting_both = ((customer_metrics['visits_per_month'] >= ideal_visits) & 
                                  (customer_metrics['spending_per_month'] >= ideal_spend)).sum()
        
        col1.metric("Meeting Frequency Target", 
                   f"{customers_meeting_freq} ({customers_meeting_freq/len(customer_metrics)*100:.1f}%)")
        col2.metric("Meeting Spending Target",
                   f"{customers_meeting_spend} ({customers_meeting_spend/len(customer_metrics)*100:.1f}%)")
        col3.metric("Meeting Both Targets",
                   f"{customers_meeting_both} ({customers_meeting_both/len(customer_metrics)*100:.1f}%)")
        col4.metric("Avg Loyalty Score", f"{customer_metrics['loyalty_score'].mean():.1f}")
        
        st.divider()
        
        # ================================================================
        # WEEKDAY VS WEEKEND ANALYSIS
        # ================================================================
        
        st.header("📅 Weekday vs Weekend Shopping Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weekend_revenue = df_filtered[df_filtered['is_weekend']][amount_col].sum()
            weekday_revenue = df_filtered[~df_filtered['is_weekend']][amount_col].sum()
            weekend_txns = len(df_filtered[df_filtered['is_weekend']])
            weekday_txns = len(df_filtered[~df_filtered['is_weekend']])
            
            fig = go.Figure(data=[
                go.Bar(name='Revenue (KES)', x=['Weekday', 'Weekend'], y=[weekday_revenue, weekend_revenue]),
                go.Bar(name='Transactions', x=['Weekday', 'Weekend'], y=[weekday_txns, weekend_txns], yaxis='y2')
            ])
            
            fig.update_layout(
                title='Weekday vs Weekend Performance',
                yaxis=dict(title='Revenue (KES)'),
                yaxis2=dict(title='Transactions', overlaying='y', side='right'),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not shopping_patterns.empty:
                shopper_dist = shopping_patterns['shopper_type'].value_counts()
                
                fig = px.pie(values=shopper_dist.values, names=shopper_dist.index,
                           title='Customer Shopping Preference',
                           color_discrete_map={
                               '🌴 Weekend Shopper': '#FF6B6B',
                               '📊 Balanced Shopper': '#4ECDC4',
                               '💼 Weekday Shopper': '#45B7D1'
                           })
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Day of week breakdown
        st.subheader("📊 Revenue by Day of Week")
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_revenue = df_filtered.groupby('day_of_week')[amount_col].sum().reindex(day_order).reset_index()
        day_revenue.columns = ['Day', 'Revenue']
        
        fig = px.bar(day_revenue, x='Day', y='Revenue', title='Revenue Distribution by Day',
                    labels={'Revenue': 'Revenue (KES)'}, 
                    color='Revenue', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ================================================================
        # VISIT PATTERN ANALYSIS
        # ================================================================
        
        st.header("🕐 Customer Visit Pattern Analysis")
        
        st.markdown("""
        Understand **when** your customers typically visit - specific days, times, and frequency patterns.
        """)
        
        if not visit_patterns.empty:
            # Merge with customer metrics to show patterns for top customers
            top_20_customers = customer_metrics.head(20)[customer_col].tolist()
            patterns_top = visit_patterns[visit_patterns[customer_col].isin(top_20_customers)].copy()
            
            if not patterns_top.empty:
                # Merge to get loyalty info
                patterns_display = patterns_top.merge(
                    customer_metrics[[customer_col, 'loyalty_score', 'loyalty_tier', 'level']],
                    on=customer_col
                )
                
                patterns_display = patterns_display[[customer_col, 'pattern', 'total_visits', 
                                                    'avg_days_between', 'loyalty_score', 
                                                    'loyalty_tier', 'level']].copy()
                
                patterns_display['avg_days_between'] = patterns_display['avg_days_between'].apply(
                    lambda x: f"{x:.1f} days" if pd.notna(x) else "N/A"
                )
                patterns_display['loyalty_score'] = patterns_display['loyalty_score'].apply(lambda x: f"{x:.0f}")
                
                patterns_display.columns = ['Customer', 'Visit Pattern', 'Total Visits', 
                                           'Avg Days Between', 'Loyalty Score', 'Tier', 'Level']
                
                st.subheader("🏆 Top 20 Customers - Visit Patterns")
                st.dataframe(patterns_display, use_container_width=True, hide_index=True, height=400)
            
            # Hour distribution
            st.subheader("⏰ Popular Shopping Hours")
            
            hour_dist = df_filtered['hour'].value_counts().sort_index()
            hour_df = pd.DataFrame({'Hour': hour_dist.index, 'Visits': hour_dist.values})
            hour_df['Time'] = hour_df['Hour'].apply(lambda x: f"{x%12 or 12}:00 {'AM' if x < 12 else 'PM'}")
            
            fig = px.bar(hour_df, x='Time', y='Visits', title='Visit Distribution by Hour of Day',
                        labels={'Visits': 'Number of Visits'}, color='Visits',
                        color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Day frequency heatmap
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Most Common Visit Days")
                day_dist = df_filtered['day_of_week'].value_counts().reindex(day_order)
                day_df = pd.DataFrame({'Day': day_dist.index, 'Visits': day_dist.values})
                
                fig = px.bar(day_df, x='Day', y='Visits', title='Total Visits by Day',
                           color='Visits', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Repeat customer patterns
                st.subheader("🔄 Visit Frequency Patterns")
                
                if 'avg_days_between' in visit_patterns.columns:
                    freq_categories = []
                    for days in visit_patterns['avg_days_between'].dropna():
                        if days <= 7:
                            freq_categories.append('Weekly (≤7 days)')
                        elif days <= 14:
                            freq_categories.append('Bi-weekly (8-14 days)')
                        elif days <= 30:
                            freq_categories.append('Monthly (15-30 days)')
                        else:
                            freq_categories.append('Occasional (>30 days)')
                    
                    if freq_categories:
                        freq_dist = pd.Series(freq_categories).value_counts()
                        
                        fig = px.pie(values=freq_dist.values, names=freq_dist.index,
                                   title='Customer Visit Frequency Distribution')
                        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ================================================================
        # LOYALTY TIER & LEVEL ANALYSIS
        # ================================================================
        
        st.header("⭐ Loyalty Tier & Level Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tier_dist = customer_metrics['loyalty_tier'].value_counts()
            
            fig = px.pie(values=tier_dist.values, names=tier_dist.index,
                        title='Customer Distribution by Tier',
                        color_discrete_map={
                            '🏆 VIP Champions': '#FFD700',
                            '⭐ Loyal Customers': '#C0C0C0',
                            '💎 Growing Customers': '#CD7F32',
                            '🌱 New/Casual Customers': '#90EE90'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            level_dist = customer_metrics['level'].value_counts()
            
            fig = px.pie(values=level_dist.values, names=level_dist.index,
                        title='Customer Distribution by Level',
                        color_discrete_map={
                            '🔥 Highest Level': '#FF4500',
                            '⚡ Mid-Level': '#FFA500',
                            '🌟 Beginner Level': '#FFD700'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("📊 Tier Statistics")
            
            for tier in ['🏆 VIP Champions', '⭐ Loyal Customers', '💎 Growing Customers', '🌱 New/Casual Customers']:
                if tier in customer_metrics['loyalty_tier'].values:
                    tier_data = customer_metrics[customer_metrics['loyalty_tier'] == tier]
                    tier_revenue = tier_data['total_spent'].sum()
                    tier_count = len(tier_data)
                    
                    st.metric(
                        tier,
                        f"{tier_count} customers",
                        f"{format_currency(tier_revenue)}"
                    )
        
        st.divider()
        
        # ================================================================
        # INSIGHTS & RECOMMENDATIONS
        # ================================================================
        
        st.header("💡 Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Findings")
            
            # Top performers
            top_10_revenue = customer_metrics.head(10)['total_spent'].sum()
            top_10_pct = top_10_revenue / total_revenue * 100
            
            st.success(f"""
            **Top 10 Customers**
            - Generate **{format_currency(top_10_revenue)}** ({top_10_pct:.1f}% of total revenue)
            - Average loyalty score: **{customer_metrics.head(10)['loyalty_score'].mean():.1f}**
            - These are your VIP customers! 🌟
            """)
            
            # Customers close to ideal
            almost_ideal = customer_metrics[
                ((customer_metrics['visits_per_month'] >= ideal_visits * 0.7) & 
                 (customer_metrics['spending_per_month'] >= ideal_spend * 0.7)) &
                ((customer_metrics['visits_per_month'] < ideal_visits) | 
                 (customer_metrics['spending_per_month'] < ideal_spend))
            ]
            
            if len(almost_ideal) > 0:
                st.info(f"""
                **{len(almost_ideal)} Customers Near Ideal Benchmark**
                - Within 70% of ideal frequency or spending
                - **High potential for conversion!**
                - Small nudge could make them ideal customers
                """)
            
            # Beginner level opportunities
            beginners = customer_metrics[customer_metrics['level'] == '🌟 Beginner Level']
            if len(beginners) > 0:
                beginner_revenue = beginners['total_spent'].sum()
                st.warning(f"""
                **{len(beginners)} Beginner Level Customers**
                - Current revenue: {format_currency(beginner_revenue)}
                - **Growth opportunity:** Nurture these customers!
                - Could increase frequency and spending
                """)
        
        with col2:
            st.subheader("🚀 Action Plan")
            
            st.markdown("""
            **For 🔥 Highest Level Customers:**
            - ✅ Exclusive VIP rewards program
            - ✅ Personal account managers
            - ✅ Early access to new products
            - ✅ Special event invitations
            
            **For ⚡ Mid-Level Customers:**
            - 📈 Targeted promotions to increase frequency
            - 🎯 Personalized product recommendations
            - 💳 Loyalty points for next tier
            - 📧 Regular engagement emails
            
            **For 🌟 Beginner Level Customers:**
            - 🎁 Welcome offers and incentives
            - 📞 Follow-up after first purchase
            - 🎓 Education about products/services
            - 💬 Feedback surveys to understand needs
            
            **Based on Visit Patterns:**
            - 🌴 Weekend shoppers: Weekend-specific deals
            - 💼 Weekday shoppers: Lunch-hour promotions
            - 🕐 Peak hour optimization: Staff accordingly
            """)
        
        st.divider()
        
        # ================================================================
        # EXPORT DATA
        # ================================================================
        
        st.header("📥 Export Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full customer analysis
            export_data = customer_metrics[[customer_col, 'visit_count', 'visits_per_month',
                                           'total_spent', 'spending_per_month', 'frequency_score',
                                           'spending_score', 'loyalty_score', 'rank',
                                           'loyalty_tier', 'level']].copy()
            
            export_data.columns = ['Customer', 'Total Visits', 'Visits per Month', 'Total Spent',
                                  'Spending per Month', 'Frequency Score', 'Spending Score',
                                  'Loyalty Score', 'Rank', 'Tier', 'Level']
            
            csv_full = export_data.to_csv(index=False)
            
            st.download_button(
                "📊 Download Full Analysis",
                csv_full,
                f"customer_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Top customers only
            top_50 = customer_metrics.head(50)[[customer_col, 'loyalty_score', 'rank', 
                                                'loyalty_tier', 'level', 'total_spent']].copy()
            csv_top = top_50.to_csv(index=False)
            
            st.download_button(
                "🏆 Download Top 50 Customers",
                csv_top,
                f"top_customers_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            # Visit patterns
            if not visit_patterns.empty:
                patterns_export = visit_patterns.merge(
                    customer_metrics[[customer_col, 'loyalty_score', 'level']],
                    on=customer_col
                )
                csv_patterns = patterns_export.to_csv(index=False)
                
                st.download_button(
                    "🕐 Download Visit Patterns",
                    csv_patterns,
                    f"visit_patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()