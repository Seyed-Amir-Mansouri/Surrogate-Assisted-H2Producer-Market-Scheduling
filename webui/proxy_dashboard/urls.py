from django.urls import path

from . import views

app_name = "proxy_dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("run/", views.run, name="run"),
    path("country/<str:country>/", views.country_detail, name="country_detail"),
]
