from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Profile(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE,null=True)
    phone_number=models.CharField(max_length=10)
    gender=models.CharField(max_length=30)

from django.db import models

class RealEstate(models.Model):
    location=models.CharField(max_length=400)
    size=models.CharField(max_length=400)
    sqft=models.CharField(max_length=400)
    bhk=models.CharField(max_length=400)
    bath=models.CharField(max_length=400)
    price=models.CharField(max_length=400)
    area_type=models.CharField(max_length=400) 
    


class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    message = models.TextField()

    def __str__(self):
        return self.name


class Feedback(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    rating = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)])
    message = models.TextField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.rating}★)"
