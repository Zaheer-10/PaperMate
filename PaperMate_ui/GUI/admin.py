import math
from .models import Paper
from .models import RecentPaper
from django.contrib import admin
from django.db import transaction


@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    """
    Admin interface configuration for the 'Paper' model.
    """
    # Display these fields in the admin list view.
    list_display = ('title', 'abstract', 'terms', 'url', 'ids')
    
    # Add a custom action for removing duplicates.
    actions = ['remove_duplicates']

    @transaction.atomic
    def remove_duplicates(self, request, queryset):
        seen_titles = set()
        duplicates = []
        batch_size = 1000  
        total_duplicates = queryset.count()
        batches = math.ceil(total_duplicates / batch_size)

        for paper in queryset:
            if paper.title in seen_titles:
                duplicates.append(paper)
            else:
                seen_titles.add(paper.title)

        for i in range(batches):
            batch_duplicates = duplicates[i * batch_size: (i + 1) * batch_size]
            batch_pks = [paper.pk for paper in batch_duplicates]
            Paper.objects.filter(pk__in=batch_pks).delete()

        self.message_user(request, f"Deleted {total_duplicates} duplicate records.")

    remove_duplicates.short_description = "Remove selected duplicates"



# Register the 'RecentPaper' model with the default admin interface.
admin.site.register(RecentPaper)
