"""
Customer Loyalty Dashboard - Frequency-Weighted Rating System
Prioritizes repeat customers over one-time big spenders
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Customer Loyalty Dashboard",
    page_icon="⭐",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(uploaded_file):
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
    try:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        
        if series.dtype == 'object':
            series = series.astype(str)
            series = series.str.replace('$', '', regex=False)
            series = series.str.replace(',', '', regex=False)
            series = series.str.strip()
        
        return pd.to_numeric(series, errors='coerce')
    except:
        return pd.to_numeric(series, errors='coerce')

def prepare_data(df, customer_col, date_col, amount_col):
    try:
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        df_clean[date_col] = safe_to_datetime(df_clean[date_col])
        df_clean[amount_col] = safe_to_numeric(df_clean[amount_col])
        df_clean = df_clean.dropna(subset=[customer_col, date_col, amount_col])
        df_clean = df_clean[df_clean[amount_col] > 0]
        df_clean = df_clean.drop_duplicates()
        
        if len(df_clean) == 0:
            st.error("No valid data after cleaning!")
            return None
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            st.info(f"🧹 Cleaned data: {rows_removed} invalid rows removed ({initial_rows} → {len(df_clean)} rows)")
        
        return df_clean
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def calculate_loyalty_scores(df, customer_col, date_col, amount_col, freq_weight=0.6, spend_weight=0.4):
    """
    Calculate customer loyalty scores with frequency weighted higher
    
    Parameters:
    freq_weight: Weight for frequency (default 60%)
    spend_weight: Weight for spending (default 40%)
    """
    try:
        # Calculate metrics per customer
        customer_metrics = df.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            date_col: ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = [customer_col, 'total_spent', 'avg_spent', 'visit_count', 'first_visit', 'last_visit']
        
        # Calculate frequency score (0-100)
        # Higher visit count = higher score
        max_visits = customer_metrics['visit_count'].max()
        min_visits = customer_metrics['visit_count'].min()
        
        if max_visits == min_visits:
            customer_metrics['frequency_score'] = 100
        else:
            customer_metrics['frequency_score'] = ((customer_metrics['visit_count'] - min_visits) / 
                                                   (max_visits - min_visits) * 100)
        
        # Calculate spending score (0-100)
        max_spent = customer_metrics['total_spent'].max()
        min_spent = customer_metrics['total_spent'].min()
        
        if max_spent == min_spent:
            customer_metrics['spending_score'] = 100
        else:
            customer_metrics['spending_score'] = ((customer_metrics['total_spent'] - min_spent) / 
                                                  (max_spent - min_spent) * 100)
        
        # Calculate weighted loyalty score
        customer_metrics['loyalty_score'] = (
            customer_metrics['frequency_score'] * freq_weight +
            customer_metrics['spending_score'] * spend_weight
        )
        
        # Calculate recency (days since last visit)
        current_date = pd.to_datetime(df[date_col].max()) + timedelta(days=1)
        customer_metrics['days_since_last_visit'] = (current_date - pd.to_datetime(customer_metrics['last_visit'])).dt.days
        
        # Assign loyalty tier
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
        
        # Sort by loyalty score
        customer_metrics = customer_metrics.sort_values('loyalty_score', ascending=False)
        
        return customer_metrics
    
    except Exception as e:
        st.error(f"Error calculating loyalty scores: {e}")
        return pd.DataFrame()

def create_customer_comparison_chart(customer_metrics, customer_col):
    """Create a scatter plot comparing frequency vs spending"""
    try:
        fig = go.Figure()
        
        # Color by loyalty tier
        tier_colors = {
            '🏆 VIP Champions': '#FFD700',
            '⭐ Loyal Customers': '#C0C0C0',
            '💎 Growing Customers': '#CD7F32',
            '🌱 New/Casual Customers': '#90EE90'
        }
        
        for tier in customer_metrics['loyalty_tier'].unique():
            tier_data = customer_metrics[customer_metrics['loyalty_tier'] == tier]
            
            fig.add_trace(go.Scatter(
                x=tier_data['visit_count'],
                y=tier_data['total_spent'],
                mode='markers',
                name=tier,
                text=tier_data[customer_col],
                marker=dict(
                    size=tier_data['loyalty_score'] / 5,  # Size based on score
                    color=tier_colors.get(tier, '#808080'),
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Visits: %{x}<br>' +
                             'Total Spent: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Customer Frequency vs Spending (Bubble Size = Loyalty Score)',
            xaxis_title='Number of Visits (Frequency)',
            yaxis_title='Total Amount Spent ($)',
            hovermode='closest',
            showlegend=True,
            height=500
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating comparison chart: {e}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⭐ Customer Loyalty Dashboard")
    st.markdown("""
    **Frequency-First Rating System** - Rewards repeat customers over one-time big spenders!
    
    Upload your data to see:
    - 🏆 Customer loyalty rankings (frequency-weighted)
    - 📊 Spending patterns per customer
    - 🔄 Visit frequency analysis
    - ⭐ Smart customer tiers
    """)
    
    # Sidebar
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    with st.sidebar.expander("📋 Required Format"):
        st.markdown("""
        Your file needs:
        - **Customer** ID or Name
        - **Date** of transaction
        - **Amount** spent
        
        Example:
        ```
        customer_id, date, amount
        C001, 2024-01-15, 150.00
        C002, 2024-01-16, 200.00
        ```
        """)
    
    # Rating weights
    st.sidebar.header("⚙️ Rating Weights")
    st.sidebar.markdown("Adjust how much each factor matters:")
    
    freq_weight = st.sidebar.slider(
        "Frequency Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="How much weight to give visit frequency (default 60%)"
    )
    
    spend_weight = 1.0 - freq_weight
    
    st.sidebar.info(f"""
    **Current Weights:**
    - Frequency: **{freq_weight*100:.0f}%**
    - Spending: **{spend_weight*100:.0f}%**
    """)
    
    if uploaded_file is None:
        st.info("👆 Upload your customer data to begin")
        
        # Show example scenario
        st.subheader("📝 Example Scenario")
        st.markdown("""
        **Who's the better customer?**
        
        With **frequency-first rating (60% frequency, 40% spending)**:
        """)
        
        example = pd.DataFrame({
            'Customer': ['Customer A', 'Customer B'],
            'Visits': [1, 8],
            'Total Spent': ['$100,000', '$90,000'],
            'Frequency Score': [0, 100],
            'Spending Score': [100, 90],
            'Loyalty Score': [40, 96],
            'Rating': ['🌱 New/Casual', '🏆 VIP Champion']
        })
        
        st.dataframe(example, use_container_width=True, hide_index=True)
        
        st.success("✅ Customer B wins with 96 loyalty score vs 40!")
        st.markdown("**Why?** Repeat customers are more valuable - they're proven, loyal, and likely to continue!")
        
        # Sample data
        st.divider()
        st.subheader("📊 Sample Data Format")
        sample = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C001', 'C003', 'C002', 'C001'],
            'customer_name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Jane Smith', 'John Doe'],
            'date': ['2024-01-15', '2024-01-16', '2024-02-10', '2024-02-15', '2024-03-01', '2024-03-15'],
            'amount': [150.00, 200.00, 175.00, 5000.00, 180.00, 220.00]
        })
        
        st.dataframe(sample, use_container_width=True)
        
        csv = sample.to_csv(index=False)
        st.download_button("📥 Download Sample", csv, "sample_data.csv", "text/csv")
        return
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        return
    
    st.success(f"✅ Loaded {len(df):,} rows!")
    
    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Column mapping
    st.sidebar.header("🔧 Configure Columns")
    columns = df.columns.tolist()
    
    default_customer = next((c for c in columns if any(w in c.lower() for w in ['customer', 'client', 'name', 'id'])), columns[0])
    default_date = next((c for c in columns if any(w in c.lower() for w in ['date', 'time'])), columns[1] if len(columns) > 1 else columns[0])
    default_amount = next((c for c in columns if any(w in c.lower() for w in ['amount', 'price', 'revenue', 'total'])), columns[-1])
    
    customer_col = st.sidebar.selectbox("Customer Column", columns, index=columns.index(default_customer))
    date_col = st.sidebar.selectbox("Date Column", columns, index=columns.index(default_date))
    amount_col = st.sidebar.selectbox("Amount Column", columns, index=columns.index(default_amount))
    
    if len(set([customer_col, date_col, amount_col])) < 3:
        st.sidebar.error("⚠️ Select different columns!")
        return
    
    if st.sidebar.button("🚀 Analyze Data", type="primary"):
        df_clean = prepare_data(df, customer_col, date_col, amount_col)
        
        if df_clean is None or len(df_clean) == 0:
            return
        
        st.success(f"✅ Ready! {len(df_clean):,} transactions from {df_clean[customer_col].nunique():,} customers")
        
        # Calculate loyalty scores
        with st.spinner("Calculating customer loyalty scores..."):
            customer_metrics = calculate_loyalty_scores(df_clean, customer_col, date_col, amount_col, freq_weight, spend_weight)
        
        if customer_metrics.empty:
            return
        
        # ====================================================================
        # KEY METRICS
        # ====================================================================
        st.header("📊 Business Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = customer_metrics['total_spent'].sum()
        total_customers = len(customer_metrics)
        avg_visits = customer_metrics['visit_count'].mean()
        avg_spent_per_customer = customer_metrics['total_spent'].mean()
        
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Total Customers", f"{total_customers:,}")
        col3.metric("Avg Visits/Customer", f"{avg_visits:.1f}")
        col4.metric("Avg Spent/Customer", f"${avg_spent_per_customer:,.2f}")
        
        st.divider()
        
        # ====================================================================
        # LOYALTY TIER DISTRIBUTION
        # ====================================================================
        st.header("⭐ Customer Loyalty Tiers")
        
        tier_counts = customer_metrics['loyalty_tier'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title='Customer Distribution by Loyalty Tier',
                color_discrete_map={
                    '🏆 VIP Champions': '#FFD700',
                    '⭐ Loyal Customers': '#C0C0C0',
                    '💎 Growing Customers': '#CD7F32',
                    '🌱 New/Casual Customers': '#90EE90'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Tier Breakdown")
            for tier in ['🏆 VIP Champions', '⭐ Loyal Customers', '💎 Growing Customers', '🌱 New/Casual Customers']:
                if tier in tier_counts.index:
                    count = tier_counts[tier]
                    pct = count / total_customers * 100
                    
                    # Get tier stats
                    tier_data = customer_metrics[customer_metrics['loyalty_tier'] == tier]
                    tier_revenue = tier_data['total_spent'].sum()
                    tier_revenue_pct = tier_revenue / total_revenue * 100
                    
                    st.metric(
                        tier,
                        f"{count} customers ({pct:.1f}%)",
                        f"${tier_revenue:,.0f} revenue ({tier_revenue_pct:.1f}%)"
                    )
        
        st.divider()
        
        # ====================================================================
        # TOP CUSTOMERS BY LOYALTY SCORE
        # ====================================================================
        st.header("🏆 Top Customers by Loyalty Score")
        st.markdown(f"**Rating System:** {freq_weight*100:.0f}% Frequency + {spend_weight*100:.0f}% Spending")
        
        top_n = st.slider("Show top N customers", 5, 50, 20, key="top_slider")
        top_customers = customer_metrics.head(top_n).copy()
        
        # Create detailed table
        display_top = top_customers[[customer_col, 'visit_count', 'total_spent', 'avg_spent', 
                                     'frequency_score', 'spending_score', 'loyalty_score', 'loyalty_tier']].copy()
        
        display_top['total_spent'] = display_top['total_spent'].apply(lambda x: f"${x:,.2f}")
        display_top['avg_spent'] = display_top['avg_spent'].apply(lambda x: f"${x:,.2f}")
        display_top['frequency_score'] = display_top['frequency_score'].apply(lambda x: f"{x:.1f}")
        display_top['spending_score'] = display_top['spending_score'].apply(lambda x: f"{x:.1f}")
        display_top['loyalty_score'] = display_top['loyalty_score'].apply(lambda x: f"{x:.1f}")
        
        display_top.columns = ['Customer', 'Visits', 'Total Spent', 'Avg/Visit', 
                               'Freq Score', 'Spend Score', 'Loyalty Score', 'Tier']
        
        st.dataframe(display_top, use_container_width=True, hide_index=True)
        
        # Bar chart
        fig = px.bar(
            top_customers.head(15),
            x=customer_col,
            y='loyalty_score',
            color='loyalty_tier',
            title=f'Top 15 Customers by Loyalty Score',
            labels={customer_col: 'Customer', 'loyalty_score': 'Loyalty Score'},
            color_discrete_map={
                '🏆 VIP Champions': '#FFD700',
                '⭐ Loyal Customers': '#C0C0C0',
                '💎 Growing Customers': '#CD7F32',
                '🌱 New/Casual Customers': '#90EE90'
            },
            hover_data={'visit_count': True, 'total_spent': ':,.2f'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ====================================================================
        # FREQUENCY VS SPENDING COMPARISON
        # ====================================================================
        st.header("📊 Frequency vs Spending Analysis")
        
        comparison_fig = create_customer_comparison_chart(customer_metrics, customer_col)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        st.markdown("""
        **How to read this chart:**
        - **X-axis (Frequency):** Number of visits
        - **Y-axis (Spending):** Total amount spent
        - **Bubble size:** Loyalty score (bigger = higher loyalty)
        - **Colors:** Loyalty tiers
        
        💡 **Insight:** Customers in the upper-right corner (high frequency + high spending) are your absolute best!
        """)
        
        st.divider()
        
        # ====================================================================
        # DETAILED CUSTOMER ANALYSIS
        # ====================================================================
        st.header("🔍 Individual Customer Deep Dive")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            selected_tier = st.multiselect(
                "Filter by Tier",
                options=customer_metrics['loyalty_tier'].unique(),
                default=customer_metrics['loyalty_tier'].unique()
            )
        
        with col2:
            min_visits = st.number_input("Minimum Visits", min_value=1, value=1)
        
        # Apply filters
        filtered = customer_metrics[
            (customer_metrics['loyalty_tier'].isin(selected_tier)) &
            (customer_metrics['visit_count'] >= min_visits)
        ]
        
        st.info(f"Showing {len(filtered)} customers matching filters")
        
        # Detailed table with all metrics
        detailed_display = filtered[[customer_col, 'visit_count', 'total_spent', 'avg_spent', 
                                    'days_since_last_visit', 'frequency_score', 'spending_score', 
                                    'loyalty_score', 'loyalty_tier']].copy()
        
        detailed_display['total_spent'] = detailed_display['total_spent'].apply(lambda x: f"${x:,.2f}")
        detailed_display['avg_spent'] = detailed_display['avg_spent'].apply(lambda x: f"${x:,.2f}")
        detailed_display['frequency_score'] = detailed_display['frequency_score'].apply(lambda x: f"{x:.0f}")
        detailed_display['spending_score'] = detailed_display['spending_score'].apply(lambda x: f"{x:.0f}")
        detailed_display['loyalty_score'] = detailed_display['loyalty_score'].apply(lambda x: f"{x:.0f}")
        
        detailed_display.columns = ['Customer', 'Visits', 'Total Spent', 'Avg/Visit', 
                                   'Days Since Last Visit', 'Freq Score', 'Spend Score', 
                                   'Loyalty Score', 'Tier']
        
        st.dataframe(detailed_display, use_container_width=True, hide_index=True, height=400)
        
        st.divider()
        
        # ====================================================================
        # INSIGHTS & RECOMMENDATIONS
        # ====================================================================
        st.header("💡 Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Findings")
            
            # VIP champions stats
            vip_champions = customer_metrics[customer_metrics['loyalty_tier'] == '🏆 VIP Champions']
            if len(vip_champions) > 0:
                vip_revenue = vip_champions['total_spent'].sum()
                vip_pct = len(vip_champions) / total_customers * 100
                vip_revenue_pct = vip_revenue / total_revenue * 100
                
                st.success(f"""
                **VIP Champions ({len(vip_champions)} customers, {vip_pct:.1f}%)**
                - Generate **${vip_revenue:,.0f}** ({vip_revenue_pct:.1f}% of revenue)
                - Average **{vip_champions['visit_count'].mean():.1f}** visits each
                - These are your BEST customers! 🌟
                """)
            
            # One-time big spenders
            one_timers = customer_metrics[customer_metrics['visit_count'] == 1]
            if len(one_timers) > 0:
                one_time_pct = len(one_timers) / total_customers * 100
                one_time_big = one_timers[one_timers['total_spent'] > customer_metrics['total_spent'].median()]
                
                if len(one_time_big) > 0:
                    st.warning(f"""
                    **One-Time Big Spenders**
                    - **{len(one_time_big)}** customers spent a lot but only visited once
                    - **Opportunity:** Convert them to repeat customers!
                    - Strategy: Follow-up, loyalty program, special offers
                    """)
            
            # Frequent small spenders
            frequent = customer_metrics[customer_metrics['visit_count'] >= 5]
            if len(frequent) > 0:
                st.info(f"""
                **Frequent Visitors ({len(frequent)} customers)**
                - Visit **5+ times**
                - Average spend: **${frequent['avg_spent'].mean():,.2f}** per visit
                - **Opportunity:** Upsell to increase transaction value!
                """)
        
        with col2:
            st.subheader("🚀 Action Items")
            
            st.markdown("""
            **For VIP Champions (Top 20%):**
            - ✅ Create exclusive VIP program
            - ✅ Offer early access to new products
            - ✅ Personal thank you messages
            - ✅ Special discounts/rewards
            
            **For One-Time Big Spenders:**
            - 📧 Send follow-up emails
            - 🎁 Offer "welcome back" discount
            - 📞 Personal call to understand needs
            - 💳 Introduce loyalty program
            
            **For Growing Customers:**
            - 📈 Encourage more frequent visits
            - 🎯 Targeted promotions
            - ⭐ Gamify loyalty points
            
            **For New/Casual Customers:**
            - 📣 Re-engagement campaigns
            - 🎁 First purchase incentives
            - 📊 Survey to understand barriers
            """)
        
        st.divider()
        
        # ====================================================================
        # EXPORT DATA
        # ====================================================================
        st.header("📥 Export Analysis")
        
        # Prepare export data
        export_data = customer_metrics[[customer_col, 'visit_count', 'total_spent', 'avg_spent',
                                       'frequency_score', 'spending_score', 'loyalty_score', 
                                       'loyalty_tier', 'days_since_last_visit']].copy()
        
        export_data.columns = ['Customer', 'Visits', 'Total Spent', 'Avg Per Visit',
                              'Frequency Score', 'Spending Score', 'Loyalty Score',
                              'Tier', 'Days Since Last Visit']
        
        csv_export = export_data.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📊 Download Full Customer Analysis",
                csv_export,
                f"customer_loyalty_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export VIP list only
            vip_export = customer_metrics[customer_metrics['loyalty_tier'] == '🏆 VIP Champions'][[customer_col, 'visit_count', 'total_spent', 'loyalty_score']].copy()
            vip_csv = vip_export.to_csv(index=False)
            
            st.download_button(
                "🏆 Download VIP Champions List",
                vip_csv,
                f"vip_champions_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()