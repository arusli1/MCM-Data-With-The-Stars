import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load SHAP importance for success
df = pd.read_csv("results/shap_importance_success.csv")

# Filter for partners only
partners = df[df['feature'].str.contains("partner_clean_", case=False)].copy()

# Sort by signed SHAP (descending)
partners = partners.sort_values('signed_shap', ascending=False)

# Clean names
partners['partner_name'] = partners['feature_clean'].str.replace("Partner: ", "")

# Create diverging bar chart
plt.figure(figsize=(12, 10))

colors = ['#2ca02c' if x > 0 else '#d62728' for x in partners['signed_shap']]

# Create boolean column for consistent hue mapping
partners['is_positive'] = partners['signed_shap'] >= 0

sns.barplot(
    data=partners,
    x='signed_shap',
    y='partner_name',
    hue='is_positive',
    palette={True: '#2ca02c', False: '#d62728'},
    hue_order=[True, False],
    legend=False
)

plt.axvline(0, color='black', linewidth=1)
plt.title("Pro Partner Impact on Success (SHAP Values)", fontsize=16, fontweight='bold')
plt.xlabel("SHAP Impact (Left=Hurts, Right=Helps)", fontsize=12)
plt.ylabel("")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results/pro_shap_rankings.png", dpi=300)
plt.close()

# Save CSV
partners[['partner_name', 'signed_shap', 'mean_abs_shap']].to_csv("results/pro_shap_rankings.csv", index=False)

print("SHAP-based Partner Rankings:")
print(partners[['partner_name', 'signed_shap']].to_string(index=False))
