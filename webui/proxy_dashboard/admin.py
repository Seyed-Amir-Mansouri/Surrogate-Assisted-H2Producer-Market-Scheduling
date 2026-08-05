from django.contrib import admin

from .models import RunResult


@admin.register(RunResult)
class RunResultAdmin(admin.ModelAdmin):
    list_display = ('country', 'elec_zone', 'h2_zone', 'start_day', 'end_day', 'created_at')
    list_filter = ('country',)
    ordering = ('-created_at',)
