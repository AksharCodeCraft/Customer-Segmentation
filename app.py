from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import requests
import os
from customer_segmentation import CustomerSegmentation

app = Flask(__name__)

# Configure AI API
AI_API_KEY = "nutKFjXorQ2V39ngVPGQSpxuFm4Nr2jRG63tn2am"
AI_API_URL = "https://api.openai.com/v1/chat/completions"

def ensure_analysis_results():
    """Ensure analysis results exist by running the analysis if necessary."""
    required_files = [
        "rfm_analysis_results.csv",
        "behavioral_analysis_results.csv",
        "deep_learning_analysis_results.csv"
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        # Initialize segmentation
        segmentation = CustomerSegmentation()
        
        # Load and process data
        segmentation.load_data()
        
        # Perform analyses
        segmentation.perform_rfm_analysis()
        segmentation.create_behavioral_segments()
        segmentation.perform_deep_learning_analysis()
        
        # Save results
        segmentation.save_results()

def load_segment_data():
    try:
        # Ensure analysis results exist
        ensure_analysis_results()
        
        # Load the results
        rfm_data = pd.read_csv("rfm_analysis_results.csv")
        behavioral_data = pd.read_csv("behavioral_analysis_results.csv")
        deep_learning_data = pd.read_csv("deep_learning_analysis_results.csv")
        
        return {
            "rfm": rfm_data["RFM_Segment"].value_counts().to_dict(),
            "behavioral": behavioral_data["Behavioral_Segment"].value_counts().to_dict(),
            "deepLearning": deep_learning_data["Deep_Learning_Segment"].value_counts().to_dict(),
            "rfm_segments": rfm_data.to_dict(orient="records"),
            "behavioral_segments": behavioral_data.to_dict(orient="records"),
            "deep_learning_segments": deep_learning_data.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_ai_recommendations(segment_type, segment_id):
    try:
        # Load the relevant data based on segment type
        if segment_type == "rfm":
            data = pd.read_csv("rfm_analysis_results.csv")
            segment_data = data[data["RFM_Segment"] == segment_id]
            metrics = {
                "Recency": segment_data["Recency"].mean(),
                "Frequency": segment_data["Frequency"].mean(),
                "Monetary": segment_data["Monetary"].mean(),
                "RFM_Score": segment_data["RFM_Score"].mean()
            }
        elif segment_type == "behavioral":
            data = pd.read_csv("behavioral_analysis_results.csv")
            segment_data = data[data["Behavioral_Segment"] == int(segment_id)]
            metrics = {
                "Page_Views": segment_data["Page Views"].mean(),
                "Time_on_Site": segment_data["Time on Site (minutes)"].mean(),
                "Purchase_Frequency": segment_data["Purchase Frequency"].mean(),
                "Avg_Spend": segment_data["Avg Spend Per Purchase"].mean()
            }
        else:
            data = pd.read_csv("deep_learning_analysis_results.csv")
            segment_data = data[data["Deep_Learning_Segment"] == int(segment_id)]
            metrics = {
                "Purchase_Mean": segment_data["Purchase_Mean"].mean(),
                "Page_Views": segment_data["Page Views"].mean(),
                "Time_on_Site": segment_data["Time on Site (minutes)"].mean(),
                "Cart_Abandonment": segment_data["Cart Abandonment Rate"].mean()
            }

        # Define recommendation strategies based on segment characteristics
        recommendation_strategies = {
            "High": {
                "customer_type": "Premium Customer",
                "shopping_style": "You are one of our most valued customers with high purchase frequency and value",
                "personalized_approach": "We will provide you with premium, personalized service",
                "actions": [
                    "Access to exclusive VIP offers",
                    "Early access to new products",
                    "Priority customer service",
                    "Special event invitations"
                ],
                "special_features": [
                    "VIP shopping experience",
                    "Personal shopping assistant",
                    "Premium rewards program",
                    "Exclusive member benefits"
                ]
            },
            "Medium-High": {
                "customer_type": "Growing Value Customer",
                "shopping_style": "You show great potential with increasing engagement",
                "personalized_approach": "We will help you discover more premium benefits",
                "actions": [
                    "Unlock premium features",
                    "Explore loyalty rewards",
                    "Discover personalized deals",
                    "Join special programs"
                ],
                "special_features": [
                    "Enhanced shopping tools",
                    "Loyalty tier benefits",
                    "Personalized recommendations",
                    "Special promotions"
                ]
            },
            "Medium": {
                "customer_type": "Regular Customer",
                "shopping_style": "You are a consistent shopper with regular engagement",
                "personalized_approach": "We will enhance your shopping experience",
                "actions": [
                    "Explore new products",
                    "Join loyalty program",
                    "Get personalized deals",
                    "Earn shopping rewards"
                ],
                "special_features": [
                    "Smart recommendations",
                    "Regular offers",
                    "Shopping guides",
                    "Member benefits"
                ]
            },
            "Medium-Low": {
                "customer_type": "Occasional Shopper",
                "shopping_style": "You shop with us occasionally",
                "personalized_approach": "We will show you more of what you might like",
                "actions": [
                    "Discover new items",
                    "Get special offers",
                    "Find similar products",
                    "Save favorites"
                ],
                "special_features": [
                    "Easy navigation",
                    "Product suggestions",
                    "Special discounts",
                    "Simplified checkout"
                ]
            },
            "Low": {
                "customer_type": "New or Returning Customer",
                "shopping_style": "We would love to see more of you",
                "personalized_approach": "Let us find what works best for you",
                "actions": [
                    "Get welcome offers",
                    "Explore popular items",
                    "Save your preferences",
                    "Try new arrivals"
                ],
                "special_features": [
                    "Quick start guide",
                    "Popular picks",
                    "Easy categories",
                    "Help center"
                ]
            }
        }

        # Get recommendations for the segment
        segment_recommendations = recommendation_strategies.get(str(segment_id), recommendation_strategies["Medium"])

        # Format the recommendations in a user-friendly way
        recommendations = f"""
Welcome, {segment_recommendations["customer_type"]}! 👋

Your Shopping Profile:
• {segment_recommendations["shopping_style"]}
• {segment_recommendations["personalized_approach"]}

Personalized Features for You:
{chr(10).join("• " + feature for feature in segment_recommendations["special_features"])}

Recommended Actions:
{chr(10).join("• " + action for action in segment_recommendations["actions"])}

Your Segment Metrics:
{chr(10).join("• " + key.replace("_", " ").title() + f": {value:.2f}" for key, value in metrics.items())}

We are Here to Help! 💫
Our recommendations are personalized just for you. Try these suggestions to make your shopping experience even better!
"""

        return {
            "recommendations": recommendations,
            "metrics": {
                "segment_size": len(segment_data),
                "key_metrics": metrics
            }
        }

    except Exception as e:
        return {
            "error": f"Error generating recommendations: {str(e)}",
            "recommendations": "We are currently unable to generate personalized recommendations. Please try again later."
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analysis")
def get_analysis():
    data = load_segment_data()
    if data is None:
        return jsonify({"error": "Failed to load segment data"}), 500
    return jsonify(data)

@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    try:
        data = request.json
        segment_type = data.get("segment_type")
        segment_id = data.get("segment_id")
        
        if not segment_type or not segment_id:
            return jsonify({"error": "Missing segment type or ID"}), 400
        
        recommendations = get_ai_recommendations(segment_type, segment_id)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True) 