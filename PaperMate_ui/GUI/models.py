from django.db import models
import csv
from pathlib import Path
from django.db import models
from django.utils import timezone
class Paper(models.Model):
    title = models.CharField(max_length=110)
    abstract = models.TextField()
    terms = models.CharField(max_length=20)
    url = models.URLField()
    ids = models.CharField(max_length=20) 
    # Adjust the max length as needed
    
    @classmethod
    def populate_database(cls):
        # Check if there are any existing records
        if cls.objects.exists():
            print("Data already populated.")
            return
        
        csv_file_path = Path(r"C:\Users\soulo\MACHINE_LEARNING\PaperMate\data\Filtered_arxiv_papers.csv")

        # Check if the CSV file exists
        if csv_file_path.exists():
            with open(csv_file_path, 'r' , encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cls.objects.create(
                        title=row['titles'],
                        abstract=row['abstracts'],
                        terms=row['terms'],
                        url=row['urls'],
                        ids=row['ids']
                    )
            print("Data populated successfully.")
        else:
            print("CSV file not found")
    
# class Feedback(models.Model):
    # rating = models.IntegerField(choices=[(1, "😕"), (2, "😐"), (3, "🙂")])
    # comments = models.TextField(blank=True)


class RecentPaper(models.Model):
    title = models.CharField(max_length=110)
    category = models.CharField(max_length=20)  # Category like 'Machine Learning', 'NLP', 'AI', 'Computer Vision'
    link = models.URLField()
    authors = models.CharField(max_length=200)
    published_date = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return self.title