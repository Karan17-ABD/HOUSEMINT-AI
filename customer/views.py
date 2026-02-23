from django.shortcuts import render
from django.contrib.auth.models import User,auth
from .models import Profile
from django.http import HttpResponseRedirect
from django.contrib import messages
from .models import RealEstate
from django.shortcuts import redirect, get_object_or_404


# Create your views here.
def openpage(request):
    return render(request,"openpage.html")

from django.shortcuts import render
import pandas as pd
import numpy as np
from .models import RealEstate
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def index(request):
    # --- Load dataset ---
    df_raw = pd.read_csv("static/Bengaluru_House_Data_Enriched D.csv")
    df_raw.dropna(subset=['total_sqft_num', 'bath', 'bhk', 'price', 'location', 'avg_price_per_sqft', 'area_type'], inplace=True)
    
    # --- Unique dropdown options ---
    all_locations = sorted(df_raw['location'].dropna().unique())
    all_area_types = sorted(df_raw['area_type'].dropna().unique())

    # --- Area type weight adjustments ---
    area_type_weights = {
        "Super built-up  Area": 1.0,
        "Built-up  Area": 1.05,
        "Plot  Area": 1.2,
        "Carpet  Area": 1.15
    }

    # --- Market factor per location (2024-25 trend) ---
    # This can be updated manually or later automated
    market_factor = {
        "Basavangudi": 1.2,
        "Indiranagar": 1.3,
        "Whitefield": 1.15,
        # Add other locations here...
    }

    if request.method == "POST":
        loc = request.POST['location'].strip()
        sqft = float(request.POST['sqft'])
        bath = int(request.POST['bath'])
        bhk = int(request.POST['bhk'])
        area_type = request.POST.get('area_type', '').strip()

        # --- Prepare model data ---
        df_model = df_raw[['total_sqft_num', 'bath', 'bhk', 'price', 'location', 'avg_price_per_sqft', 'area_type']].copy()
        dummies = pd.get_dummies(df_model['location'])
        X = pd.concat([df_model[['total_sqft_num', 'bath', 'bhk']], dummies], axis=1)
        y = df_model['price']

        # --- Train XGBoost ---
        xgb = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, tree_method="hist"
        )
        xgb.fit(X, y)

        # --- Prepare input for prediction ---
        x_input = pd.DataFrame([[sqft, bath, bhk] + [0]*len(dummies.columns)],
                               columns=['total_sqft_num', 'bath', 'bhk'] + list(dummies.columns))
        if loc in dummies.columns:
            x_input[loc] = 1

        pred_price = float(xgb.predict(x_input)[0])

        # --- Get location average price per sqft ---
        if loc in df_raw['location'].values:
            avg_price_per_sqft = df_raw[df_raw['location'] == loc]['avg_price_per_sqft'].mean()
        else:
            avg_price_per_sqft = df_raw['avg_price_per_sqft'].mean()

        # --- Apply area type and market factor ---
        area_weight = area_type_weights.get(area_type, 1.0)
        predicted_pps = avg_price_per_sqft * area_weight
        factor = market_factor.get(loc, 1.0)
        predicted_pps *= factor

        # --- Compute final price ---
        final_price = predicted_pps * sqft

        # --- Format price for display ---
        if final_price >= 10000000:  # >= 1 Crore
            final_price_display = f"{round(final_price/10000000,2)} Cr"
        else:
            final_price_display = f"{round(final_price/100000,1)} Lakh"

        # --- Save to DB ---
        RealEstate.objects.create(
            location=loc,
            size=f"{bhk} BHK",
            sqft=str(sqft),
            bhk=str(bhk),
            bath=str(bath),
            price=str(round(final_price/100000, 2)),  # Save in Lakhs
            area_type=area_type
        )

        return render(request, "predict.html", {
            "location": loc,
            "bhk": bhk,
            "sqft": sqft,
            "bath": bath,
            "area_type": area_type,
            "price_formatted": final_price_display,
            "price_per_sqft": round(predicted_pps, 2),
            "locations": all_locations,
            "area_types": all_area_types,
        })

    # --- Initial GET page ---
    return render(request, "index.html", {
        "locations": all_locations,
        "area_types": all_area_types,
    })


    '''if request.method=="POST":
        loc=request.POST['location']
        #size=int(request.POST[''])
        sqft1=float(request.POST['sqft'])
        bath1=int(request.POST["bath"])
        bhk1=int(request.POST['bhk'])
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        loc1=l.fit_transform([loc])
        import pandas as pd
        import numpy as np
        from matplotlib import pyplot as plt
        #get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib
        matplotlib.rcParams["figure.figsize"]=(20,10)


        # In[6]:


        df1 = pd.read_csv("static/Bengaluru_House_Data (1).csv")
        print(df1.head())


        # In[8]:


        print(df1.shape)


        # In[9]:


        print(df1.groupby('area_type')['area_type'].agg('count'))


        # In[12]:


        df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
        print(df2.head())


        # In[13]:


        print(df2.isnull().sum())


        # In[14]:


        df3 = df2.dropna()
        print(df3.isnull().sum())


        # In[17]:


        print(df3.shape)


        # 

        # In[21]:


        df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


        # In[22]:


        print(df3.head())

        # In[23]:


        print(df3['bhk'].unique())


        # In[24]:


        #df3[df3.bhk>20]


        # In[25]:


        #df3.total_sqft.unique()


        # In[26]:


        def is_float(x):
            try:
                float(x)
            except:
                return False
            return True


        # In[29]:


        #df3[~df3['total_sqft'].apply(is_float)].head(10)


        # In[30]:


        def convert_sqrt_to_num(x):
            tokens = x.split('-')
            if len(tokens) == 2:
                return(float(tokens[0])+float(tokens[1]))/2
            try:
                return float(x)
            except:
                return None


        # In[31]:


        #convert_sqrt_to_num('2166')


        # In[32]:


        print(convert_sqrt_to_num('2100 - 2850'))


        # In[33]:


        df4 = df3.copy()
        df4['total_sqft'] = df4['total_sqft'].apply(convert_sqrt_to_num)
        print(df4.head())


        # In[34]:


        #df4.loc[30]


        # In[36]:


        df5 = df4.copy()
        df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
        print(df5.head())


        # In[37]:


        print(df5.location.unique())


        # In[38]:


        len(df5.location.unique())


        # In[40]:


        df5.location = df5.location.apply(lambda x: x.strip())

        location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
        print(location_stats)


        # In[41]:


        print(len(location_stats[location_stats<=10]))


        # In[46]:


        location_stats_less_than_10 = location_stats[location_stats<=10]
        print(location_stats_less_than_10)


        # In[43]:


        print(len(df5.location.unique()))


        # In[47]:


        df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        print(len(df5.location.unique()))


        # In[48]:


        print(df5.head(10))


        # In[49]:


        #outlier removal
        print(df5[df5.total_sqft/df5.bhk<300].head())


        # In[50]:


        print(df5.shape)


        # In[51]:


        df6=df5[~(df5.total_sqft/df5.bhk<300)]
        print(df6.shape)


        # In[52]:


        print(df6.price_per_sqft.describe())


        # In[56]:


        def remove_pps_outliers(df):
            df_out = pd.DataFrame()
            for key,subdf in df.groupby('location'):
                m = np.mean(subdf.price_per_sqft)
                st = np.std(subdf.price_per_sqft)
                reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
                df_out = pd.concat([df_out,reduced_df],ignore_index=True)
            return df_out

        df7 = remove_pps_outliers(df6)
        #df7.shape


        # In[65]:


        def plot_scatter_chart(df,location):
            bhk2 = df[(df.location==location) & (df.bhk==2)]
            bhk3 = df[(df.location==location) & (df.bhk==3)]
            matplotlib.rcParams['figure.figsize'] = (15,10)
            plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
            plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
            plt.xlabel("Total Square Feet Area")
            plt.ylabel("Price Per Square Feet")
            plt.title(location)
            plt.legend()
            
        plot_scatter_chart(df7,"Hebbal")


        # In[63]:


        def remove_bhk_outliers(df):
            exclude_indices = np.array([])
            for location, location_df in df.groupby('location'):
                bhk_stats = {}
                for bhk, bhk_df in location_df.groupby('bhk'):
                    bhk_stats[bhk] = {
                        'mean': np.mean(bhk_df.price_per_sqft),
                        'std': np.std(bhk_df.price_per_sqft),
                        'count': bhk_df.shape[0]
                    }
                    for bhk, bhk_df in location_df.groupby('bhk'):
                        stats = bhk_stats.get(bhk-1)
                        if stats and stats['count']>5:
                            exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
            return df.drop(exclude_indices,axis='index')

        df8 = remove_bhk_outliers(df7)
        print(df8.shape)

                        


        # In[66]:


        plot_scatter_chart(df8,"Hebbal")


        # In[70]:


        import matplotlib
        matplotlib.rcParams["figure.figsize"] = (20,10)
        plt.hist(df8.price_per_sqft,rwidth=0.8)
        plt.xlabel("price Per Square Feet")
        plt.ylabel("count")
        plt.show()


        # In[71]:


        print(df8[df8.bath>10])


        # In[72]:


        plt.hist(df8.bath,rwidth=0.8)
        plt.xlabel("Number Of Bathrooms")
        plt.ylabel("count")
        plt.show()


        # In[73]:


        print(df8[df8.bath>df8.bhk+2])


        # In[74]:


        df9 = df8[df8.bath<df8.bhk+2]
        print(df9.shape)


        # In[75]:


        df10=df9.drop(['size','price_per_sqft'],axis='columns')
        print(df10.head(3))


        # In[78]:


        dummies=pd.get_dummies(df10.location)
        print(dummies.head(3))


        # In[79]:


        df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
        print(df11.head(3))
                    


        # In[80]:


        df12=df11.drop('location',axis='columns')
        print(df12.head(2))


        # In[81]:


        print(df12.shape)


        # In[96]:


        X=df12.drop('price',axis='columns')
        print(X.head())


        # In[83]:


        y=df12.price
        print(y.head())


        # In[97]:


        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


        # In[98]:


        from sklearn.linear_model import LinearRegression
        lr_clf = LinearRegression()
        lr_clf.fit(X_train,y_train)
        lr_clf.score(X_test,y_test)


        # In[99]:


        from sklearn.model_selection import ShuffleSplit
        from sklearn.model_selection import cross_val_score

        cv= ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        cross_val_score(LinearRegression(),X,y, cv=cv)


        # In[110]:


        def predict_price(location,sqft,bath,bhk):
            loc_index = np.where(X.columns==location)[0][0]
            
            x = np.zeros(len(X.columns))
            x[0] =sqft
            x[1] =bath
            x[2] =bhk
            if loc_index >=0:
                x[loc_index]=1
                
            return lr_clf.predict([x])[0]


        # In[114]:


        print(predict_price('1st Phase JP Nagar',1000, 2, 3))


        # In[113]:


        print(predict_price('Indira Nagar',1000, 2, 2))
        print(loc1,sqft1,bath1,bhk1)
        import numpy as np
        price=np.array([[loc1,sqft1,bath1,bhk1]],dtype=object)
        print(price)
        prediction=predict_price([[loc1,sqft1,bath1,bhk1]])
        return render(request,"predict.html",{"location":loc,"sqft":sqft1,"bath":bath1,"bhk":bhk1,"prediction":prediction})
    return render(request,"index.html")'''

    

def about(request):
    return render(request,"about.html")

def house(request):
    return render(request,"house.html")


def contact(request):
    return render(request,"contact.html")


def price(request):
    return render(request,"price.html")

def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')


def login(request):
    if request.method=="POST":
        em=request.POST['uname']
        psw=request.POST['psw']
        user=auth.authenticate(username=em,password=psw)
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('/index')
        else:
            messages.info(request,"Invalid email or Password")
            return render(request,"login.html")
    return render(request,"login.html")

def register1(request):
    if request.method=="POST":
        fn=request.POST['fname']
        un=request.POST['uname']
        em=request.POST['email']
        phn=request.POST['phn']
        psw=request.POST['psw']
        psw1=request.POST['psw1']
        gender=request.POST['gender']
        if psw==psw1:
            if User.objects.filter(username=un).exists():
                messages.info(request,"username Already Exists")
                return render(request,"register1.html")
            elif User.objects.filter(email=em).exists():
                messages.info(request,"Email Already Exists")
                return render(request,"register1.html")
            else:
                #store value in database
                #Create object for the table name
                 user=User.objects.create_user(first_name=fn,email=em,username=un,password=psw)
            Profile.objects.create(user=user,phone_number=phn,gender=gender)
            user.save()
            return HttpResponseRedirect('login')
        else:
             messages.info(request,"Password Not Matching")
             return render(request,"register1.html")
    else:
         return render(request,"register1.html")
    
from django.shortcuts import render
import pandas as pd
import numpy as np
from .models import RealEstate
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def data(request):
    # --- Load dataset ---
    df_raw = pd.read_csv("static/Bengaluru_House_Data_Enriched D.csv")
    df_raw.dropna(subset=['total_sqft_num', 'bath', 'bhk', 'price', 'location', 'avg_price_per_sqft', 'area_type'], inplace=True)
    
    # --- Unique dropdown options ---
    all_locations = sorted(df_raw['location'].dropna().unique())
    all_area_types = sorted(df_raw['area_type'].dropna().unique())

    # --- Area type weight adjustments ---
    area_type_weights = {
        "Super built-up  Area": 1.0,
        "Built-up  Area": 1.05,
        "Plot  Area": 1.2,
        "Carpet  Area": 1.15
    }

    # --- Market factor per location (2024-25 trend) ---
    # This can be updated manually or later automated
    market_factor = {
        "Basavangudi": 1.2,
        "Indiranagar": 1.3,
        "Whitefield": 1.15,
        # Add other locations here...
    }

    if request.method == "POST":
        loc = request.POST['location'].strip()
        sqft = float(request.POST['sqft'])
        bath = int(request.POST['bath'])
        bhk = int(request.POST['bhk'])
        area_type = request.POST.get('area_type', '').strip()

        # --- Prepare model data ---
        df_model = df_raw[['total_sqft_num', 'bath', 'bhk', 'price', 'location', 'avg_price_per_sqft', 'area_type']].copy()
        dummies = pd.get_dummies(df_model['location'])
        X = pd.concat([df_model[['total_sqft_num', 'bath', 'bhk']], dummies], axis=1)
        y = df_model['price']

        # --- Train XGBoost ---
        xgb = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, tree_method="hist"
        )
        xgb.fit(X, y)

        # --- Prepare input for prediction ---
        x_input = pd.DataFrame([[sqft, bath, bhk] + [0]*len(dummies.columns)],
                               columns=['total_sqft_num', 'bath', 'bhk'] + list(dummies.columns))
        if loc in dummies.columns:
            x_input[loc] = 1

        pred_price = float(xgb.predict(x_input)[0])

        # --- Get location average price per sqft ---
        if loc in df_raw['location'].values:
            avg_price_per_sqft = df_raw[df_raw['location'] == loc]['avg_price_per_sqft'].mean()
        else:
            avg_price_per_sqft = df_raw['avg_price_per_sqft'].mean()

        # --- Apply area type and market factor ---
        area_weight = area_type_weights.get(area_type, 1.0)
        predicted_pps = avg_price_per_sqft * area_weight
        factor = market_factor.get(loc, 1.0)
        predicted_pps *= factor

        # --- Compute final price ---
        final_price = predicted_pps * sqft

        # --- Format price for display ---
        if final_price >= 10000000:  # >= 1 Crore
            final_price_display = f"{round(final_price/10000000,2)} Cr"
        else:
            final_price_display = f"{round(final_price/100000,1)} Lakh"

        # --- Save to DB ---
        RealEstate.objects.create(
            location=loc,
            size=f"{bhk} BHK",
            sqft=str(sqft),
            bhk=str(bhk),
            bath=str(bath),
            price=str(round(final_price/100000, 2)),  # Save in Lakhs
            area_type=area_type
        )

        return render(request, "predict.html", {
            "location": loc,
            "bhk": bhk,
            "sqft": sqft,
            "bath": bath,
            "area_type": area_type,
            "price_formatted": final_price_display,
            "price_per_sqft": round(predicted_pps, 2),
            "locations": all_locations,
            "area_types": all_area_types,
        })

    # --- Initial GET page ---
    return render(request, "data.html", {
        "locations": all_locations,
        "area_types": all_area_types,
    })

    
def predict(request):
    return render(request, "predict.html")

def adminlogin(request):
    if request.method == "POST":
        if request.method == "POST":
            usid = request.POST['uname']
            pswd = request.POST['psw']
            if usid == 'admin' and pswd == 'admin':
                return HttpResponseRedirect('adminhome')

    return render(request,"adminlogin.html")

def adminhome(request):
    real_estate=RealEstate.objects.all()
    return render(request,"adminhome.html",{"real_estate":real_estate})

def delete_realestate(request, id):
    if request.method == 'POST':
        obj = get_object_or_404(RealEstate, pk=id)
        obj.delete()
    return redirect('adminhome')  # Replace with your main view's name

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import difflib
import re

@csrf_exempt
def chatbot(request):
    df = pd.read_csv("static/Bengaluru_House_Data (1).csv")
    df.dropna(subset=['location', 'size', 'price', 'total_sqft', 'area_type'], inplace=True)
    df['bhk'] = df['size'].apply(lambda x: int(re.search(r'\d+', x).group()) if pd.notna(x) and re.search(r'\d+', x) else 0)
    df['total_sqft'] = df['total_sqft'].apply(lambda x: (sum(map(float, str(x).split('-')))/2 if '-' in str(x) else pd.to_numeric(x, errors='coerce')))
    df.dropna(subset=['total_sqft'], inplace=True)
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    area_type_weights = {
        "Super built-up  Area": 1.0,
        "Built-up  Area": 1.05,
        "Carpet  Area": 1.15,
        "Plot  Area": 1.2
    }

    def adjust_price(location_df):
        base_price = location_df['price'].mean()
        avg_pps = location_df['price_per_sqft'].mean()
        weighted = []
        for area, factor in area_type_weights.items():
            subset = location_df[location_df['area_type'] == area]
            if not subset.empty:
                weighted.append(subset['price'].mean() * factor)
        adjusted_price = np.mean(weighted) if weighted else base_price
        return round(adjusted_price, 2), round(avg_pps, 2)

    # Load chat history from session or initialize
    chat_history = request.session.get('chat_history', [])

    if request.method == "POST":
        user_input = request.POST.get("user_input", "").strip()
        user_input_lower = user_input.lower()
        response = "Sorry, I didn't understand that. Please try rephrasing your question."

        location_list = df['location'].unique()
        matched_location = None
        for loc in location_list:
            if loc.lower() in user_input_lower:
                matched_location = loc
                break
        if not matched_location:
            matches = difflib.get_close_matches(user_input_lower, [loc.lower() for loc in location_list], n=1, cutoff=0.6)
            if matches:
                matched_location = next((loc for loc in location_list if loc.lower() == matches[0]), None)

        # Handle compare queries
        if "compare" in user_input_lower and "and" in user_input_lower:
            try:
                parts = user_input_lower.split("and")
                loc1 = parts[0].replace("compare", "").strip()
                loc2 = parts[1].strip()
                loc_match1 = difflib.get_close_matches(loc1, [loc.lower() for loc in location_list], n=1, cutoff=0.6)
                loc_match2 = difflib.get_close_matches(loc2, [loc.lower() for loc in location_list], n=1, cutoff=0.6)
                if loc_match1 and loc_match2:
                    loc1_real = next(loc for loc in location_list if loc.lower() == loc_match1[0])
                    loc2_real = next(loc for loc in location_list if loc.lower() == loc_match2[0])
                    adj1, _ = adjust_price(df[df['location'] == loc1_real])
                    adj2, _ = adjust_price(df[df['location'] == loc2_real])
                    response = f"Average adjusted price:\n• {loc1_real}: ₹{adj1} Lakhs\n• {loc2_real}: ₹{adj2} Lakhs"
                else:
                    response = "Couldn't clearly match both locations. Try again."
            except:
                response = "Error parsing locations for comparison."

        elif matched_location:
            df_loc = df[df['location'] == matched_location]
            adj_price, adj_pps = adjust_price(df_loc)

            if re.search(r'\b(2|3)\s*(bhk|bedroom)', user_input_lower):
                bhk_query = int(re.search(r'\b(2|3)', user_input_lower).group())
                bhk_data = df_loc[df_loc['bhk'] == bhk_query]
                if not bhk_data.empty:
                    response = f"Yes, {bhk_query} BHK homes are available in {matched_location}, starting from ₹{round(bhk_data['price'].min(), 2)} Lakhs."
                else:
                    response = f"No {bhk_query} BHK homes found in {matched_location}."

            elif "most expensive" in user_input_lower or "max price" in user_input_lower:
                max_price = df_loc['price'].max()
                response = f"The most expensive house in {matched_location} costs around ₹{round(max_price, 2)} Lakhs."

            elif "average price" in user_input_lower:
                response = f"The adjusted average price in {matched_location} is ₹{adj_price} Lakhs (market aligned)."

        # Save chat
        chat_history.append((user_input, response))
        request.session['chat_history'] = chat_history
        return render(request, "chatbot.html", {"chat_history": chat_history})

    return render(request, "chatbot.html", {"chat_history": chat_history})

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages

def reset_password(request):
    if request.method == "POST":
        username = request.POST.get("username")
        new_pass = request.POST.get("new_password")
        confirm_pass = request.POST.get("confirm_password")

        if new_pass != confirm_pass:
            messages.error(request, "Passwords do not match.")
            return redirect('reset_password')

        try:
            user = User.objects.get(username=username)
            user.set_password(new_pass)
            user.save()
            messages.success(request, "Password updated successfully. You can now login.")
            return render(request, "login.html")  # ✅ Changed here
        except User.DoesNotExist:
            messages.error(request, "Username not found.")
            return redirect('reset_password')

    return render(request, "reset_password.html")


from django.shortcuts import render, redirect
from .models import Contact

def save_contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        message = request.POST.get('message')

        Contact.objects.create(name=name, email=email, phone=phone, message=message)

        return redirect('thank_you')  # Optional page or redirect
    return redirect('home')  # fallback

from .models import Feedback
from django.shortcuts import render, redirect

def feedback_view(request):
    if request.method == 'POST':
        Feedback.objects.create(
            name=request.POST['name'],
            email=request.POST['email'],
            rating=request.POST['rating'],
            message=request.POST['message']
        )
        return redirect('feedback_thankyou')  # Create a thank you page or redirect back
    return render(request, 'feedback.html')

def admin_contact_data(request):
    contacts = Contact.objects.all().order_by('-id')
    return render(request, 'admin_contact_data.html', {'contacts': contacts})

def admin_feedback_data(request):
    feedbacks = Feedback.objects.all().order_by('-submitted_at')
    return render(request, 'admin_feedback_data.html', {'feedbacks': feedbacks})

def feedback_thankyou(request):
    return render(request, 'feedback_thankyou.html')


