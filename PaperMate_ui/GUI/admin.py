from django.db import transaction
from django.contrib import admin
from .models import Paper
import math

@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = ('title', 'abstract', 'terms', 'url', 'ids')
    actions = ['remove_duplicates']

    @transaction.atomic
    def remove_duplicates(self, request, queryset):
        seen_titles = set()
        duplicates = []
        batch_size = 1000  # Adjust the batch size as needed
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
