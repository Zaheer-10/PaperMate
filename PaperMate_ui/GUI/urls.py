from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', views.index, name='index'),
    path('qa/', views.qa_page, name='qa'),
    path('recommendations/' , views.recommendations, name='recommendations'),
    path('about/', views.about, name='about'),
    path('search/', views.search_papers, name='search_papers'),
    # path('<str:category>/', views.index, name='index_category'),
    # path('recommendation/', views.recommendation, name='recommendation'),
    ]
# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)