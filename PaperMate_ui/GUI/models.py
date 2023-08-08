from django.db import models
import csv
from pathlib import Path
from django.db import models

class Paper(models.Model):
    title = models.CharField(max_length=110)
    abstract = models.TextField()
    terms = models.CharField(max_length=20)
    url = models.URLField()
    ids = models.CharField(max_length=20) 
    # Adjust the max length as needed
    
    @classmethod
    def populate_database(cls):
        csv_file = Path()
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cls.objects.create(
                    title=row['tittles'],
                    abstract=row['abstract'],
                    terms=row['terms'],
                    url=row['url'],
                    ids=row['ids']
                )
# class Feedback(models.Model):
    # rating = models.IntegerField(choices=[(1, "😕"), (2, "😐"), (3, "🙂")])
    # comments = models.TextField(blank=True)
