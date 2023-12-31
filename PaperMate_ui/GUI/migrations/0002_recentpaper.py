# Generated by Django 4.1.2 on 2023-08-11 12:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('GUI', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='RecentPaper',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=110)),
                ('category', models.CharField(max_length=20)),
                ('link', models.URLField()),
                ('authors', models.CharField(max_length=200)),
                ('published_date', models.DateTimeField()),
            ],
        ),
    ]
