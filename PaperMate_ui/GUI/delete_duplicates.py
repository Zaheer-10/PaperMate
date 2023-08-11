from django.core.management.base import BaseCommand
from GUI.models import Paper
from django.db import transaction
import math

class Command(BaseCommand):
    help = 'Deletes duplicate records from the Paper model'

    def add_arguments(self, parser):
        parser.add_argument('batch_size', type=int, default=1000, help='Batch size for deletion')

    @transaction.atomic
    def handle(self, *args, **options):
        batch_size = options['batch_size']
        total_duplicates = Paper.objects.count()
        batches = math.ceil(total_duplicates / batch_size)

        self.stdout.write(self.style.SUCCESS(f"Total duplicates: {total_duplicates}"))
        self.stdout.write(self.style.SUCCESS(f"Batch size: {batch_size}"))
        self.stdout.write(self.style.SUCCESS(f"Batches: {batches}"))

        for i in range(batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_duplicates = Paper.objects.values_list('id', flat=True).distinct('title')[batch_start:batch_end]
            Paper.objects.filter(id__in=batch_duplicates).delete()
            self.stdout.write(self.style.SUCCESS(f"Deleted batch {i + 1}/{batches}"))

        self.stdout.write(self.style.SUCCESS("Duplicate removal completed"))
