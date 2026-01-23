import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Workplace Mental Health Analysis", page_icon="🧠", layout="wide")

# Title
st.title("Worklace Mental Health")
st.write("Analysing workplace mental health")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('mental_health_dataset.csv')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filters")

# Work environment
work_env = st.sidebar.multiselect(
    "Work Environment",
    df["work_environment"].unique(),
    df["work_environment"].unique()
)

#Mental Health Risk
risk = st.sidebar.multiselect(
    "Mental Health Risk",
    df["mental_health_risk"].unique(),
    df["mental_health_risk"].unique()
)

filtered_df = df[
    (df["work_environment"].isin(work_env)) &
    (df["mental_health_risk"].isin(risk))
]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Prediction Interface", "ML Models and Comparison", "Data"])

with tab1:
    # Basic info
    st.header("Dataset Overview")
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Productivity", f"{filtered_df.productivity_score.mean():.1f}")
    c2.metric("Avg Stress", f"{filtered_df.stress_level.mean():.1f}")
    c3.metric("Avg Sleep (hrs)", f"{filtered_df.sleep_hours.mean():.1f}")
    c4.metric("High Risk %", f"{(filtered_df.mental_health_risk=='High').mean()*100:.1f}%")
    
    # Basic statistics
    st.subheader("Statistics")
    st.write(filtered_df.describe())

with tab2:
    st.header("Data Visualisations")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Productivity Distribution
        st.subheader('Average Productivity Distribution')
        fig1, ax1 = plt.subplots()
        ax1.hist(filtered_df['productivity_score'], bins=5, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Productivity')
        ax1.set_ylabel('Average')
        st.pyplot(fig1)
         
    with col2:
        # Sleep Hours distribution
        st.subheader("Sleep Pattern")
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered_df['sleep_hours'], bins=5, color='skyblue', edgecolor='black')
        ax2.set_xlabel('sleep hours')
        ax2.set_ylabel('')
        st.pyplot(fig2)

    # Risk Category distribution
    st.subheader("Risk Distribution")
    fig, ax = plt.subplots()
    risk_counts = filtered_df['mental_health_risk'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    if len(risk_counts) > 0:
        ax.pie(risk_counts, labels=['Medium', 'High', 'Low'], 
                autopct='%1.1f%%', colors=colors)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    numerical_cols = filtered_df.select_dtypes(include='number').columns.tolist()
    if len(filtered_df) > 10:
        correlation_matrix = filtered_df[numerical_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax3)
        st.pyplot(fig3)

# Machine Learning Model
with tab3:
    st.header("Random Forest Model")
    
    # Prepare data
    @st.cache_data
    def prepare_data(dataframe):
        # Copy data
        X = dataframe.drop(['mental_health_risk'], axis=1).copy()
        y = dataframe['mental_health_risk'].copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        return X, y, label_encoders
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model with Random Forest..."):
            # Prepare data
            X, y, encoders = prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['features'] = X.columns.tolist()
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.subheader('Classification Report')
                st.write(classification_report(y_test, y_pred, target_names=['Medium', 'High', 'Low']))
            
            # Feature Importance plot
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
                
            fig2, ax2 = plt.subplots()
            ax2.barh(importance_df['feature'], importance_df['importance'])
            ax2.set_xlabel('Importance')
            st.pyplot(fig2)

            st.success("Model trained successfully!")
    
    # Prediction Interface
    st.markdown("---")
    st.header("Make a Prediction")
    
    if 'model' in st.session_state:
        st.write("Enter the details below to predict Mental Health Risk:")
        
        # Create input form
        with st.form("prediction_form"):
            age_input = st.number_input(
                "Age",
                min_value=18,
                max_value=65,
                value=18,
                step=1
            )
                
            anxiety_input = st.number_input(
                "Anxiety Score",
                min_value=0,
                max_value=21,
                value=0,
                step=1
            )

            prod_input = st.number_input(
                "Productivity Score",
                min_value=40,
                max_value=100,
                value=40,
                step=1
            )

            dep_input = st.number_input(
                "Depression Score",
                min_value=0,
                max_value=30,
                value=0,
                step=1
            )

            soc_input = st.number_input(
                "Social Support",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )

            sleep_input = st.number_input(
                "Sleep Hours",
                min_value=3,
                max_value=10,
                value=3,
                step=1
            )
            
            # Submit button
            submit_button = st.form_submit_button("Predict Mental Health Risk")
        
            if submit_button:
                # Create input dataframe with all features
                # Note: You'll need to match the exact feature names from your dataset
                input_data = pd.DataFrame({
                    'age': [age_input],
                    'anxiety': [anxiety_input],
                    'productivity': [prod_input],
                    'depression': [dep_input],
                    'sleep hours': [sleep_input],
                    'social support': [soc_input]
                })
            
                # Ensure all features from training are present
                # Add any missing features with default values if needed
                for feature in st.session_state['features']:
                    if feature not in input_data.columns:
                        input_data[feature] = 0  # or appropriate default
            
                # Reorder columns to match training data
                input_data = input_data[st.session_state['features']]
            
                # Encode categorical variables if any
                for col in input_data.columns:
                    if col in st.session_state['encoders']:
                        # Handle encoding for categorical features
                        pass  # Implement if you have categorical features
            
                # Make prediction
                prediction = st.session_state['model'].predict(input_data)[0]
                prediction_proba = st.session_state['model'].predict_proba(input_data)[0]
            
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
            
                col1, col2 = st.columns(2)
            
                with col1:
                    if prediction == 2:
                        st.error("**Medium Risk**")
                        st.metric("Menatl Health Risk", f"{prediction_proba[2]:.1%}")
                    elif prediction == 1:
                        st.error("**High Risk**")
                        st.metric("Mental Health Risk", f"{prediction_proba[0]:.1%}")
                    else:
                        st.success("**Low Risk**")
                        st.metric("Mental Health Risk", f"{prediction_proba[1]:.1%}")
            
                with col2:
                    st.write("**Probability Breakdown:**")
                    st.write(f"- Medium Risk: {prediction_proba[2]:.1%}")
                    st.write(f"- High Risk: {prediction_proba[0]:.1%}")
                    st.write(f"- Low Risk: {prediction_proba[1]:.1%}")
                
                    # Visual representation
                    fig, ax = plt.subplots(figsize=(6, 2))
                    categories = ['Medium', 'High', 'Low']
                    probabilities = prediction_proba
                    colors = ['#2ecc71', '#e74c3c', '#3984db']
                    ax.barh(categories, probabilities, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    for i, v in enumerate(probabilities):
                        ax.text(v + 0.02, i, f'{v:.1%}', va='center')
                    st.pyplot(fig)

    else:
        st.warning("Please train the model first using the 'Train Model' button above.")
    
    # Display existing model info
    if 'model' in st.session_state:        
        # Show top features
        st.subheader("Top 5 Most Important Features")
        model = st.session_state['model']
        features = st.session_state['features']
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        for idx, row in importance_df.iterrows():
            st.write(f"- **{row['feature']}**: {row['importance']:.3f}")

with tab4:
    st.header("Raw Data")
    st.dataframe(filtered_df)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Workplace Mental Health:** Analysis mental health across Workplace")
