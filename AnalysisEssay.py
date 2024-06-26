import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

# Load the data
data_path = st.file_uploader("Upload your CSV file", type=["csv"])
if data_path is not None:
    data = pd.read_csv(data_path)

# Convert 'Reporting starts' to datetime
data['Reporting starts'] = pd.to_datetime(data['Reporting starts'])

# Calculate metrics for the last 3 days and 7 days
today = data['Reporting starts'].max()
last_3_days = today - timedelta(days=3)
last_7_days = today - timedelta(days=7)

data_last_3_days = data[data['Reporting starts'] >= last_3_days]
data_last_7_days = data[data['Reporting starts'] >= last_7_days]

# Define a function to calculate metrics for a given period
def calculate_metrics(data_period):
    metrics = {
        'impressions': data_period['Impressions'].sum(),
        'clicks': data_period['Link clicks'].sum(),
        'landing_page_views': data_period['Landing page views'].sum(),
        'adds_to_cart': data_period['Adds to cart'].sum(),
        'checkouts_initiated': data_period['Checkouts initiated'].sum(),
        'cost': data_period['Amount spent (INR)'].sum(),
        'conversion_value': data_period['Purchases conversion value'].sum(),
        'roas': data_period['Purchase ROAS (return on ad spend)'].mean(),
        'cpm': data_period['CPM (cost per 1,000 impressions) (INR)'].mean(),
        'ctr': data_period['CTR (link click-through rate)'].mean(),
        'cpc': data_period['CPC (cost per link click) (INR)'].mean(),
    }
    return metrics

# Calculate metrics for the last 3 days
metrics_3_days = calculate_metrics(data_last_3_days)

# Calculate metrics for the last 7 days
metrics_7_days = calculate_metrics(data_last_7_days)

# Calculate growth rates
previous_3_days = data[(data['Reporting starts'] < last_3_days) & (data['Reporting starts'] >= last_3_days - timedelta(days=3))]
previous_7_days = data[(data['Reporting starts'] < last_7_days) & (data['Reporting starts'] >= last_7_days - timedelta(days=7))]

def calculate_growth(current, previous):
    return ((current - previous) / previous) * 100 if previous != 0 else np.nan

growth_3_days = {metric: calculate_growth(metrics_3_days[metric], calculate_metrics(previous_3_days)[metric]) for metric in metrics_3_days}
growth_7_days = {metric: calculate_growth(metrics_7_days[metric], calculate_metrics(previous_7_days)[metric]) for metric in metrics_7_days}

# Generate a dynamic recommendation based on the growth trend
def dynamic_recommendation(growth):
    recommendations = []
    if growth['cost'] > 0:
        if growth['cost'] <= 10:
            recommendations.append("Cost increased slightly. Monitor closely for further changes.")
        elif 10 < growth['cost'] <= 30:
            recommendations.append("Cost increased moderately. Review targeting and optimize budget allocation.")
        elif 30 < growth['cost'] <= 50:
            recommendations.append("Cost increased significantly. Reassess targeting and campaign efficiency.")
        elif 50 < growth['cost'] <= 80:
            recommendations.append("Cost increased substantially. Implement cost-saving measures and refine targeting.")
        elif growth['cost'] == 100:
            recommendations.append("Cost doubled. Immediate action required to optimize spending and ROI.")

    elif growth['cost'] < 0:
        if growth['cost'] >= -10:
            recommendations.append("Cost decreased slightly. Monitor for sustainable trends.")
        elif -10 > growth['cost'] >= -30:
            recommendations.append("Cost decreased moderately. Review targeting and assess impact on performance.")
        elif -30 > growth['cost'] >= -50:
            recommendations.append("Cost decreased significantly. Analyze targeting efficiency and adjust accordingly.")
        elif -50 > growth['cost'] >= -80:
            recommendations.append("Cost decreased substantially. Consider reallocating budget for better results.")
        elif growth['cost'] == -100:
            recommendations.append("Cost decreased by 100%. Investigate for potential data discrepancies.")

    # Impressions Growth Recommendations
    if growth['impressions'] > 0:
        if growth['impressions'] <= 10:
            recommendations.append("Impressions increased slightly. Monitor for sustained growth.")
        elif 10 < growth['impressions'] <= 30:
            recommendations.append("Impressions increased moderately. Review targeting for enhanced reach.")
        elif 30 < growth['impressions'] <= 50:
            recommendations.append("Impressions increased significantly. Optimize ad placements for wider exposure.")
        elif 50 < growth['impressions'] <= 80:
            recommendations.append("Impressions increased substantially. Explore new audience segments for scaling.")
        elif growth['impressions'] == 100:
            recommendations.append("Impressions doubled. Capitalize on increased visibility with targeted campaigns.")

    elif growth['impressions'] < 0:
        if growth['impressions'] >= -10:
            recommendations.append("Impressions decreased slightly. Monitor closely for potential fluctuations.")
        elif -10 > growth['impressions'] >= -30:
            recommendations.append("Impressions decreased moderately. Review targeting and ad relevance.")
        elif -30 > growth['impressions'] >= -50:
            recommendations.append("Impressions decreased significantly. Optimize ad placements and targeting.")
        elif -50 > growth['impressions'] >= -80:
            recommendations.append("Impressions decreased substantially. Consider revising campaign strategies.")
        elif growth['impressions'] == -100:
            recommendations.append("Impressions decreased by 100%. Investigate for potential issues or errors.")

# Adds to Cart Growth Recommendations
    if growth['adds_to_cart'] > 0:
        if growth['adds_to_cart'] <= 10:
            recommendations.append("Adds to Cart increased slightly. Monitor for sustained growth.")
        elif 10 < growth['adds_to_cart'] <= 30:
            recommendations.append("Adds to Cart increased moderately. Review product relevance and targeting.")
        elif 30 < growth['adds_to_cart'] <= 50:
            recommendations.append("Adds to Cart increased significantly. Optimize product descriptions and visuals.")
        elif 50 < growth['adds_to_cart'] <= 80:
            recommendations.append("Adds to Cart increased substantially. Implement retargeting strategies.")
        elif growth['adds_to_cart'] == 100:
            recommendations.append("Adds to Cart doubled. Focus on converting cart additions to purchases.")

    elif growth['adds_to_cart'] < 0:
        if growth['adds_to_cart'] >= -10:
            recommendations.append("Adds to Cart decreased slightly. Monitor closely for potential fluctuations.")
        elif -10 > growth['adds_to_cart'] >= -30:
            recommendations.append("Adds to Cart decreased moderately. Review targeting and product offerings.")
        elif -30 > growth['adds_to_cart'] >= -50:
            recommendations.append("Adds to Cart decreased significantly. Optimize product visibility and incentives.")
        elif -50 > growth['adds_to_cart'] >= -80:
            recommendations.append("Adds to Cart decreased substantially. Analyze checkout flow and barriers.")
        elif growth['adds_to_cart'] == -100:
            recommendations.append("Adds to Cart decreased by 100%. Investigate for potential issues or errors.")

# Clicks Growth Recommendations
    if growth['clicks'] > 0:
        if growth['clicks'] <= 10:
            recommendations.append("Clicks increased slightly. Monitor for sustained engagement.")
        elif 10 < growth['clicks'] <= 30:
            recommendations.append("Clicks increased moderately. Review ad creatives for enhanced click-through.")
        elif 30 < growth['clicks'] <= 50:
            recommendations.append("Clicks increased significantly. Optimize targeting for higher click rates.")
        elif 50 < growth['clicks'] <= 80:
            recommendations.append("Clicks increased substantially. Implement ad testing for further improvement.")
        elif growth['clicks'] == 100:
            recommendations.append("Clicks doubled. Focus on conversion optimization strategies.")

    elif growth['clicks'] < 0:
        if growth['clicks'] >= -10:
            recommendations.append("Clicks decreased slightly. Monitor closely for potential fluctuations.")
        elif -10 > growth['clicks'] >= -30:
            recommendations.append("Clicks decreased moderately. Review targeting and ad relevance.")
        elif -30 > growth['clicks'] >= -50:
            recommendations.append("Clicks decreased significantly. Optimize ad placements and targeting.")
        elif -50 > growth['clicks'] >= -80:
            recommendations.append("Clicks decreased substantially. Consider revising campaign strategies.")
        elif growth['clicks'] == -100:
            recommendations.append("Clicks decreased by 100%. Investigate for potential issues or errors.")

    # Landing Page Views Growth Recommendations
    if growth['landing_page_views'] > 0:
        if growth['landing_page_views'] <= 10:
            recommendations.append("Landing Page Views increased slightly. Monitor for sustained traffic.")
        elif 10 < growth['landing_page_views'] <= 30:
            recommendations.append("Landing Page Views increased moderately. Optimize landing page for conversions.")
        elif 30 < growth['landing_page_views'] <= 50:
            recommendations.append("Landing Page Views increased significantly. Test different landing page layouts.")
        elif 50 < growth['landing_page_views'] <= 80:
            recommendations.append("Landing Page Views increased substantially. Implement A/B testing for optimization.")
        elif growth['landing_page_views'] == 100:
            recommendations.append("Landing Page Views doubled. Focus on improving user experience.")

    elif growth['landing_page_views'] < 0:
        if growth['landing_page_views'] >= -10:
            recommendations.append("Landing Page Views decreased slightly. Monitor closely for potential fluctuations.")
        elif -10 > growth['landing_page_views'] >= -30:
            recommendations.append("Landing Page Views decreased moderately. Review landing page relevance.")
        elif -30 > growth['landing_page_views'] >= -50:
            recommendations.append("Landing Page Views decreased significantly. Optimize landing page content.")
        elif -50 > growth['landing_page_views'] >= -80:
            recommendations.append("Landing Page Views decreased substantially. Analyze user behavior on the page.")
        elif growth['landing_page_views'] == -100:
            recommendations.append("Landing Page Views decreased by 100%. Investigate for potential issues.")
# CPM Growth Recommendations
    if growth['cpm'] > 0:
        if growth['cpm'] <= 10:
            recommendations.append("CPM increased slightly. Monitor closely for further changes.")
        elif 10 < growth['cpm'] <= 30:
            recommendations.append("CPM increased moderately. Review targeting and optimize ad placements.")
        elif 30 < growth['cpm'] <= 50:
            recommendations.append("CPM increased significantly. Consider revising ad formats or placements.")
        elif 50 < growth['cpm'] <= 80:
            recommendations.append("CPM increased substantially. Explore cost-saving measures or new ad platforms.")
        elif growth['cpm'] == 100:
            recommendations.append("CPM doubled. Reassess ad strategies and audience targeting.")

    elif growth['cpm'] < 0:
        if growth['cpm'] >= -10:
            recommendations.append("CPM decreased slightly. Monitor for sustainable trends.")
        elif -10 > growth['cpm'] >= -30:
            recommendations.append("CPM decreased moderately. Review targeting and ad placements.")
        elif -30 > growth['cpm'] >= -50:
            recommendations.append("CPM decreased significantly. Optimize ad placements and targeting.")
        elif -50 > growth['cpm'] >= -80:
            recommendations.append("CPM decreased substantially. Consider reallocating budget for better results.")
        elif growth['cpm'] == -100:
            recommendations.append("CPM decreased by 100%. Investigate for potential data discrepancies.")

    # CPC Growth Recommendations
    if growth['cpc'] > 0:
        if growth['cpc'] <= 10:
            recommendations.append("CPC increased slightly. Monitor closely for further changes.")
        elif 10 < growth['cpc'] <= 30:
            recommendations.append("CPC increased moderately. Review bidding strategies and ad relevance.")
        elif 30 < growth['cpc'] <= 50:
            recommendations.append("CPC increased significantly. Optimize ad targeting and audience segmentation.")
        elif 50 < growth['cpc'] <= 80:
            recommendations.append("CPC increased substantially. Consider testing new ad creatives and offers.")
        elif growth['cpc'] == 100:
            recommendations.append("CPC doubled. Reevaluate bidding strategies and ad performance.")

    elif growth['cpc'] < 0:
        if growth['cpc'] >= -10:
            recommendations.append("CPC decreased slightly. Monitor for sustainable trends.")
        elif -10 > growth['cpc'] >= -30:
            recommendations.append("CPC decreased moderately. Review bidding strategies and ad placements.")
        elif -30 > growth['cpc'] >= -50:
            recommendations.append("CPC decreased significantly. Optimize ad targeting and audience segmentation.")
        elif -50 > growth['cpc'] >= -80:
            recommendations.append("CPC decreased substantially. Consider reallocating budget for better results.")
        elif growth['cpc'] == -100:
            recommendations.append("CPC decreased by 100%. Investigate for potential data discrepancies.")

    # CTR Growth Recommendations
    if growth['ctr'] > 0:
        if growth['ctr'] <= 10:
            recommendations.append("CTR increased slightly. Monitor closely for further changes.")
        elif 10 < growth['ctr'] <= 30:
            recommendations.append("CTR increased moderately. Review ad creatives and calls to action.")
        elif 30 < growth['ctr'] <= 50:
            recommendations.append("CTR increased significantly. Optimize ad placements and targeting.")
        elif 50 < growth['ctr'] <= 80:
            recommendations.append("CTR increased substantially. Consider testing new ad formats.")
        elif growth['ctr'] == 100:
            recommendations.append("CTR doubled. Focus on improving ad engagement and relevance.")

    elif growth['ctr'] < 0:
        if growth['ctr'] >= -10:
            recommendations.append("CTR decreased slightly. Monitor for sustainable trends.")
        elif -10 > growth['ctr'] >= -30:
            recommendations.append("CTR decreased moderately. Review ad creatives and targeting.")
        elif -30 > growth['ctr'] >= -50:
            recommendations.append("CTR decreased significantly. Optimize ad placements and targeting.")
        elif -50 > growth['ctr'] >= -80:
            recommendations.append("CTR decreased substantially. Consider reallocating budget for better results.")
        elif growth['ctr'] == -100:
            recommendations.append("CTR decreased by 100%. Investigate for potential issues or errors.")
# Conv Value Growth Recommendations
    if growth['conversion_value'] > 0:
        if growth['conversion_value'] <= 10:
            recommendations.append("Conversion Value increased slightly. Monitor for sustained growth.")
        elif 10 < growth['conversion_value'] <= 30:
            recommendations.append("Conversion Value increased moderately. Optimize conversion funnels.")
        elif 30 < growth['conversion_value'] <= 50:
            recommendations.append("Conversion Value increased significantly. Focus on high-value audience segments.")
        elif 50 < growth['conversion_value'] <= 80:
            recommendations.append("Conversion Value increased substantially. Implement upselling or cross-selling strategies.")
        elif growth['conversion_value'] == 100:
            recommendations.append("Conversion Value doubled. Analyze factors driving higher value conversions.")

    elif growth['conversion_value'] < 0:
        if growth['conversion_value'] >= -10:
            recommendations.append("Conversion Value decreased slightly. Monitor closely for sustainable trends.")
        elif -10 > growth['conversion_value'] >= -30:
            recommendations.append("Conversion Value decreased moderately. Review conversion tracking and attribution.")
        elif -30 > growth['conversion_value'] >= -50:
            recommendations.append("Conversion Value decreased significantly. Optimize conversion funnels and targeting.")
        elif -50 > growth['conversion_value'] >= -80:
            recommendations.append("Conversion Value decreased substantially. Analyze user behavior and site performance.")
        elif growth['conversion_value'] == -100:
            recommendations.append("Conversion Value decreased by 100%. Investigate for potential issues or errors.")

    # ROAS Growth Recommendations
    if growth['roas'] > 0:
        if growth['roas'] <= 10:
            recommendations.append("ROAS increased slightly. Monitor closely for further changes.")
        elif 10 < growth['roas'] <= 30:
            recommendations.append("ROAS increased moderately. Review campaign ROI and optimize targeting.")
        elif 30 < growth['roas'] <= 50:
            recommendations.append("ROAS increased significantly. Focus on high-ROI ad placements and strategies.")
        elif 50 < growth['roas'] <= 80:
            recommendations.append("ROAS increased substantially. Consider scaling profitable campaigns.")
        elif growth['roas'] == 100:
            recommendations.append("ROAS doubled. Capitalize on high-return campaigns and audiences.")

    elif growth['roas'] < 0:
        if growth['roas'] >= -10:
            recommendations.append("ROAS decreased slightly. Monitor for sustainable trends.")
        elif -10 > growth['roas'] >= -30:
            recommendations.append("ROAS decreased moderately. Review campaign ROI and optimize targeting.")
        elif -30 > growth['roas'] >= -50:
            recommendations.append("ROAS decreased significantly. Adjust bidding and targeting strategies.")
        elif -50 > growth['roas'] >= -80:
            recommendations.append("ROAS decreased substantially. Consider revising campaign objectives and budgets.")
        elif growth['roas'] == -100:
            recommendations.append("ROAS decreased by 100%. Investigate for potential issues or errors.")
    return recommendations


# Generate essays for 3 days and 7 days
    def generate_essay(metrics, growth, period):
        growth_strings = {metric: f"{'increased' if growth[metric] >= 0 else 'dropped'} by {abs(growth[metric]):.2f}%" for metric in growth}
        recommendations = dynamic_recommendation(growth)
        recommendations_text = "\n".join([f"- {rec}" for rec in recommendations])

        essay = f"""
        Over the last {period}, the campaign has shown notable changes in key performance metrics.
        Impressions have reached {metrics['impressions']:,}, indicating a substantial increase in visibility and potential audience reach.
        Clicks have amounted to {metrics['clicks']:,}, showing increased user engagement with the content or ads.
        Landing page views were {metrics['landing_page_views']:,}, and adds to cart were {metrics['adds_to_cart']:,}, suggesting that interest has been captured to some extent.
        Checkouts initiated were {metrics['checkouts_initiated']:,}.
        
        The total cost for this period was {metrics['cost']:,.2f} INR, with a conversion value of {metrics['conversion_value']:,.2f} INR and an ROAS of {metrics['roas']:.2f}.
        CPM was {metrics['cpm']:.2f} INR, CTR was {metrics['ctr']:.2f}%, and CPC was {metrics['cpc']:.2f} INR.
        
        Compared to the previous period, impressions {growth_strings['impressions']}, clicks {growth_strings['clicks']}, landing page views {growth_strings['landing_page_views']},
        adds to cart {growth_strings['adds_to_cart']}, checkouts initiated {growth_strings['checkouts_initiated']}, and costs {growth_strings['cost']}.
        
        Recommendations:
        {recommendations_text}
        """
        return essay

    essay_3_days = generate_essay(metrics_3_days, growth_3_days, "3 days")
    essay_7_days = generate_essay(metrics_7_days, growth_7_days, "7 days")

    print("3-Day Performance Report:")
    print(essay_3_days)

    print("\n7-Day Performance Report:")
    print(essay_7_days)

    # Analyze the top and worst-performing campaign, ad, and ad set
    def performance_analysis(data, period):
        grouped_campaign = data.groupby('Campaign name').agg({
            'Amount spent (INR)': 'sum',
            'Purchases conversion value': 'sum',
            'Purchase ROAS (return on ad spend)': 'mean',
            'CPC (cost per link click) (INR)': 'mean'
        }).reset_index()
        
        grouped_ad_set = data.groupby('Ad set name').agg({
            'Amount spent (INR)': 'sum',
            'Purchases conversion value': 'sum',
            'Purchase ROAS (return on ad spend)': 'mean',
            'CPC (cost per link click) (INR)': 'mean'
        }).reset_index()
        
        grouped_ad = data.groupby('Ad name').agg({
            'Amount spent (INR)': 'sum',
            'Purchases conversion value': 'sum',
            'Purchase ROAS (return on ad spend)': 'mean',
            'CPC (cost per link click) (INR)': 'mean'
        }).reset_index()

        top_campaign = grouped_campaign.sort_values(by='Purchase ROAS (return on ad spend)', ascending=False).iloc[0]
        worst_campaign = grouped_campaign.sort_values(by='Purchase ROAS (return on ad spend)', ascending=True).iloc[0]
        top_ad_set = grouped_ad_set.sort_values(by='Purchase ROAS (return on ad spend)', ascending=False).iloc[0]
        worst_ad_set = grouped_ad_set.sort_values(by='Purchase ROAS (return on ad spend)', ascending=True).iloc[0]
        top_ad = grouped_ad.sort_values(by='Purchase ROAS (return on ad spend)', ascending=False).iloc[0]
        worst_ad = grouped_ad.sort_values(by='Purchase ROAS (return on ad spend)', ascending=True).iloc[0]

        performance_text = f"""
        Performance Analysis for {period}:
        
        Top Performing Campaign:
        Name: {top_campaign['Campaign name']}
        Amount Spent: {top_campaign['Amount spent (INR)']:.2f} INR
        Conversion Value: {top_campaign['Purchases conversion value']:.2f} INR
        ROAS: {top_campaign['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {top_campaign['CPC (cost per link click) (INR)']:.2f} INR
        
        Worst Performing Campaign:
        Name: {worst_campaign['Campaign name']}
        Amount Spent: {worst_campaign['Amount spent (INR)']:.2f} INR
        Conversion Value: {worst_campaign['Purchases conversion value']:.2f} INR
        ROAS: {worst_campaign['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {worst_campaign['CPC (cost per link click) (INR)']:.2f} INR
        
        Top Performing Ad Set:
        Name: {top_ad_set['Ad set name']}
        Amount Spent: {top_ad_set['Amount spent (INR)']:.2f} INR
        Conversion Value: {top_ad_set['Purchases conversion value']:.2f} INR
        ROAS: {top_ad_set['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {top_ad_set['CPC (cost per link click) (INR)']:.2f} INR
        
        Worst Performing Ad Set:
        Name: {worst_ad_set['Ad set name']}
        Amount Spent: {worst_ad_set['Amount spent (INR)']:.2f} INR
        Conversion Value: {worst_ad_set['Purchases conversion value']:.2f} INR
        ROAS: {worst_ad_set['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {worst_ad_set['CPC (cost per link click) (INR)']:.2f} INR
        
        Top Performing Ad:
        Name: {top_ad['Ad name']}
        Amount Spent: {top_ad['Amount spent (INR)']:.2f} INR
        Conversion Value: {top_ad['Purchases conversion value']:.2f} INR
        ROAS: {top_ad['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {top_ad['CPC (cost per link click) (INR)']:.2f} INR
        
        Worst Performing Ad:
        Name: {worst_ad['Ad name']}
        Amount Spent: {worst_ad['Amount spent (INR)']:.2f} INR
        Conversion Value: {worst_ad['Purchases conversion value']:.2f} INR
        ROAS: {worst_ad['Purchase ROAS (return on ad spend)']:.2f}
        CPC: {worst_ad['CPC (cost per link click) (INR)']:.2f} INR
        """
        
        return performance_text

    # Display performance analysis for the last 3 days and 7 days
    performance_3_days = performance_analysis(data_last_3_days, "3 days")
    performance_7_days = performance_analysis(data_last_7_days, "7 days")

    print(performance_3_days)
    print(performance_7_days)
else:
    st.write("Please upload a CSV file to proceed.")
