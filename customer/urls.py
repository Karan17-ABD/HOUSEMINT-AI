from django.urls import path 
from .import views 

urlpatterns = [
    path('openpage',views.openpage,name='openpages'),
    path('index',views.index,name='Index'),
    path('about',views.about,name='About'),
    path('house',views.house,name='House'),
    path('contact',views.contact,name='Contact'),
    path('login',views.login,name='Login'),
    path('register1',views.register1,name='Register1'),
    path('data',views.data,name='data'),
    path('predict',views.predict,name='predict'),
    path('adminlogin',views.adminlogin,name='adminlogin'),
    path('adminhome',views.adminhome,name='adminhome'),
    path('logout',views.logout,name='logout'),
    path('delete/<int:id>/', views.delete_realestate, name='delete_realestate'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('reset_password', views.reset_password, name='reset_password'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('feedback/thankyou/', views.feedback_thankyou, name='feedback_thankyou'),
    path('save-contact/', views.save_contact, name='save_contact'),
    path('admin-contact-data/', views.admin_contact_data, name='admin_contact_data'),
    path('admin-feedback-data/', views.admin_feedback_data, name='admin_feedback_data'),
]
