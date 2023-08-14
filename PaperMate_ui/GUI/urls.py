from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

# Configure URL patterns for the web application
urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('search/', views.search_papers, name='search_papers'),  # Search papers page
    path('recommendations/', views.recommendations, name='recommendations'),  # Recommendations page
    path('about/', views.about, name='about'),  # About page
    path('qa/', views.qa_page, name='qa'),  # Q&A page
    path('summarize/<str:paper_id>/', views.summarize_paper, name='summarize_paper'),


]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
