#importing the dependcies
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, roc_curve, classification_report)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2

#importing the classifier model
#with open("classifier.pkl", "rb") as pickle_in:
    #classifier = pickle.load(pickle_in)

#classifier function
def train_classifier():
    # Importing the data
    hotel_reservation_data = pd.read_csv("Hotel Reservations.csv")

    # Converting the column names to lower case
    hotel_reservation_data.columns = hotel_reservation_data.columns.str.lower()

    # Removing unnecessary columns from the data
    cols_to_drop = ['booking_id', 'arrival_year', 'arrival_date']
    hotel_reservation_data = hotel_reservation_data.drop(cols_to_drop, axis=1)

    # Create conditions for each quarter
    quarter_conditions = [
        (hotel_reservation_data['arrival_month'] <= 3),
        (hotel_reservation_data['arrival_month'] > 3) & (hotel_reservation_data['arrival_month'] <= 6),
        (hotel_reservation_data['arrival_month'] > 6) & (hotel_reservation_data['arrival_month'] <= 9),
        (hotel_reservation_data['arrival_month'] > 9)
    ]
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    # Creating a new column for quarters
    hotel_reservation_data['quarter'] = np.select(quarter_conditions, quarters)

    # Dropping the arrival month column
    hotel_reservation_data = hotel_reservation_data.drop('arrival_month', axis=1)

    # One-hot encode the categorical variables
    df = pd.get_dummies(hotel_reservation_data, drop_first=True)

    print(df.columns)

    # Preparing data for training
    X = df.drop('booking_status_Not_Canceled', axis=1)
    Y = df['booking_status_Not_Canceled']

    #Dropping the features based on the above observations
    X = X.drop(["room_type_reserved_Room_Type 2", 'room_type_reserved_Room_Type 3', 'type_of_meal_plan_Not Selected', 'type_of_meal_plan_Meal Plan 3'], axis=1)

    #Moedl training
    #Model training with 80:20 train test split and the random state is used to replicate the results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 77)

    #Performing oversampling to handle the imbalance in the prediction values
    oversampling = SMOTE()
    oversampling_X, oversampling_Y = oversampling.fit_resample(X, Y)
    oversampling_X_train, oversampling_X_test, oversampling_Y_train, over_Y_test = train_test_split(oversampling_X, oversampling_Y, test_size=0.2, stratify=oversampling_Y, random_state = 77)
    classifier = RandomForestClassifier(n_estimators=150, random_state=77) #Calcuted the optimum value using Grid Search as this process took a lot of time we have removed this code from the notebook
    classifier.fit(oversampling_X_train, oversampling_Y_train)

    # Handling imbalance in the dataset
    #oversampler = SMOTE(random_state=77)
    #X, Y = oversampler.fit_resample(X, Y)

    # Splitting the dataset
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=77)

    # Training the model
    #classifier = RandomForestClassifier(n_estimators=150, random_state=77)
    #classifier.fit(X_train, Y_train)

    return classifier

#the classifier is needed upon each app reload
classifier = train_classifier()

#Display Result function
def display_booking_status(booking_status_Not_Canceled):
    if booking_status_Not_Canceled[0]:  # Assuming the status is in a list as [True] or [False]
        # Booking won't be cancelled, display happy emoji in green color
        st.markdown(
            "<h1 style='color:green;'>Booking won't be cancelled 😊</h1>", 
            unsafe_allow_html=True
        )
    else:
        # Booking will be cancelled, display sad emoji in red color
        st.markdown(
            "<h1 style='color:red;'>Booking will be cancelled 😢</h1>", 
            unsafe_allow_html=True
        )

#function to display probabilities
def display_probabilities(not_cancellation, cancellation):
    # Round the probabilities to two decimal places
    not_cancellation = round(not_cancellation, 2)
    cancellation = round(cancellation, 2)

    # Display the probabilities with appropriate labels and colors
    st.markdown(
        f"<h4 style='color:green;'>Probability of Not Cancellation: {not_cancellation}% 😊</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h4 style='color:red;'>Probability of Cancellation: {cancellation}% 😢</h4>",
        unsafe_allow_html=True
    )




#Prediction function
def predict_booking_outcome(X):
    res = 0 #just for testing
    #column names
    feature_names = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
    'required_car_parking_space', 'lead_time', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
    'no_of_special_requests', 'type_of_meal_plan_Meal Plan 2', 'room_type_reserved_Room_Type 4',
    'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6',
    'room_type_reserved_Room_Type 7', 'market_segment_type_Complementary',
    'market_segment_type_Corporate', 'market_segment_type_Offline', 'market_segment_type_Online',
    'quarter_Q2', 'quarter_Q3', 'quarter_Q4']

    # Create a DataFrame with the feature names and the values to predict
    X_df = pd.DataFrame([X], columns=feature_names)


    prediction = classifier.predict(X_df)
    probabilities = classifier.predict_proba(X_df)

    not_cancellation = probabilities[0][1] * 100
    cancellation = probabilities[0][0] * 100

    return prediction, not_cancellation, cancellation


# Function to display the Project Overview page
def show_project_overview():
    st.header('Project Overview')
    st.write("""
    Work in progress. Information will be updated soon!
    """)

# Function to display the EDA page
def show_eda():
    st.header('Exploratory Data Analysis (EDA)')
    st.write("""
    Work in progress information will be updated soon !
    """)
    # Example of a simple line chart
    data = {'Series1': [1, 3, 4, 5, 6], 'Series2': [2, 3, 2, 4, 5]}
    st.line_chart(data)

# Function to display the Model page
def application_demo():
    st.header('Application Demo')
    with st.form("prediction_form"):
        #declaring the boolean option
        no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=99, value=0)
        no_of_children = st.number_input('Number of Children', min_value=0, max_value=99, value=0)

        no_of_weekend_nights = st.number_input('Stay Duration(Number of Days)', min_value=0, max_value=50, value=0)
        no_of_week_nights = no_of_weekend_nights

        # Parking Space start
        with st.container():
            st.markdown("""
                <style>
                    div[data-testid="stForm"] div.row-widget.stRadio > div{flex-direction:row;}
                </style>
                """, unsafe_allow_html=True)
            yes_no_options = ['Yes', 'No']
            required_car_parking_space_selection = st.radio('Parking Space Required', yes_no_options)
            required_car_parking_space = 1 if required_car_parking_space_selection == 'Yes' else 0
        #Parking space end

        lead_time = st.number_input('Days Booked in Advance', min_value=0, max_value=500, value=0)

        #Repeat guest start
        repeated_guest_selection = st.radio('Have you visited before?', yes_no_options)
        repeated_guest = 1 if repeated_guest_selection == 'Yes' else 0
        #Repeat guest end

        no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=99, value=0)
        no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=99, value=0)

        #Room cost
        avg_price_per_room = float(st.number_input('Room Price', min_value=0, max_value=1000, value=100))

        no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, max_value=10, value=0)

        #Meal Plan logic start

        # Select box for meal plan options
        st.write('Meal Plan Options')  # You can use st.write() or st.markdown() for more styling flexibility

        # Checkboxes for meal plan options
        meal_plan_1 = st.checkbox('Meal Plan 1')
        meal_plan_2 = st.checkbox('Meal Plan 2')
        meal_plan_3 = st.checkbox('Meal Plan 3')

        # Convert boolean checkbox state to integer values
        meal_plan_1 = 1 if meal_plan_1 else 0
        meal_plan_2 = 1 if meal_plan_2 else 0
        meal_plan_3 = 1 if meal_plan_3 else 0

        #Meal plan logic end

        #Room type logic start

        # Dropdown menu for room type options
        room_type_options = [f'Room Type {i}' for i in range(1, 8)]  # Dynamically create list for Room Type 1 to 7
        selected_room_type = st.selectbox('Room Type', room_type_options)

        # Initialize all room type variables to 0
        room_type_1 = room_type_2 = room_type_3 = room_type_4 = room_type_5 = room_type_6 = room_type_7 = 0

        # Set the selected room type variable to 1
        if selected_room_type == 'Room Type 1':
            room_type_1 = 1
        elif selected_room_type == 'Room Type 2':
            room_type_2 = 1
        elif selected_room_type == 'Room Type 3':
            room_type_3 = 1
        elif selected_room_type == 'Room Type 4':
            room_type_4 = 1
        elif selected_room_type == 'Room Type 5':
            room_type_5 = 1
        elif selected_room_type == 'Room Type 6':
            room_type_6 = 1
        elif selected_room_type == 'Room Type 7':
            room_type_7 = 1

        #Room type logic end

        #Booking Type logic start
        # Dropdown menu for booking mode options
        booking_mode_options = ['Online', 'Offline', 'Corporate', 'Complementary']  # Order as specified
        selected_booking_mode = st.selectbox('Booking Mode', booking_mode_options)

        # Initialize all booking mode variables to 0
        market_segment_online = market_segment_offline = market_segment_corporate = market_segment_complementary = 0

        # Set the selected booking mode variable to 1
        if selected_booking_mode == 'Online':
            market_segment_online = 1
        elif selected_booking_mode == 'Offline':
            market_segment_offline = 1
        elif selected_booking_mode == 'Corporate':
            market_segment_corporate = 1
        elif selected_booking_mode == 'Complementary':
            market_segment_complementary = 1

        #Bopking Type logic end
        
        #Booking Quarter logic start

        # Dropdown menu for quarter selection
        quarter_options = ['Q1[Jan-Mar]', 'Q2[Apr-Jun]', 'Q3[Jul-Sep]', 'Q4[Oct-Dec]']
        selected_quarter = st.selectbox('Booking Quarter', quarter_options)

        # Initialize all quarter variables to 0
        quarter_q1 = quarter_q2 = quarter_q3 = quarter_q4 = 0

        # Set the selected quarter variable to 1
        if selected_quarter == 'Q1[Jan-Mar]':
            quarter_q1 = 1
        elif selected_quarter == 'Q2[Apr-Jun]':
            quarter_q2 = 1
        elif selected_quarter == 'Q3[Jul-Sep]':
            quarter_q3 = 1
        elif selected_quarter == 'Q4[Oct-Dec]':
            quarter_q4 = 1

        #Booking Quarter logic end
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            #Passing all the data collected here
            X = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, 
                 required_car_parking_space, lead_time, repeated_guest, no_of_previous_cancellations, 
                 no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests, meal_plan_2,
                 room_type_4, room_type_5, room_type_6, room_type_7, market_segment_complementary, market_segment_corporate, 
                 market_segment_offline, market_segment_online, quarter_q2, quarter_q3, quarter_q4]
            #st.write(X) #Tested successfully
            booking_status_Not_Canceled, not_cancellation, cancellation = predict_booking_outcome(X)
            
            #displaying the result
            # Display booking status based on prediction
            display_booking_status(booking_status_Not_Canceled)
            display_probabilities(not_cancellation, cancellation)

    

# Main app logic
def main():
    st.sidebar.title('Hotel Booking Cancellations Prediction')
    # Define page navigation with default page as 'Model'
    page = st.sidebar.radio("", ['Application Demo', 'Project Overview', 'EDA'], index=0)

    if page == 'Project Overview':
        show_project_overview()
    elif page == 'EDA':
        show_eda()
    elif page == 'Application Demo':
        application_demo()

if __name__ == "__main__":
    main()
