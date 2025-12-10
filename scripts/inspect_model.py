import sys
import os
import xgboost as xgb
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def inspect_model():
    model_path = "models/xgb_direction.json"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return

    print(f"✅ Model found: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024:.2f} KB")

    try:
        bst = xgb.Booster()
        bst.load_model(model_path)
        
        print("\n📊 Model Info:")
        print(f"   Trees: {bst.num_boosted_rounds()}")
        print(f"   Features: {bst.feature_names}")
        
        print("\n🔑 Feature Importance:")
        importance = bst.get_score(importance_type='weight')
        if not importance:
            print("   ⚠️ No feature importance found (Model might be empty/untrained)")
        else:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feat, score in sorted_imp:
                print(f"   - {feat}: {score}")
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
