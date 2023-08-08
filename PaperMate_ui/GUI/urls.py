from django.urls import path
from . import views

urlpatterns = [
    #path('',views.index,name='home'),
    path("" , views.home , name='home'),
    path('qa/', views.qa_page, name='qa'),
    path('search/', views.search_papers, name='search_papers'),
    path('recommendation/', views.recommendation, name='recommendation'),
]