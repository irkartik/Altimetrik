from rest_framework import serializers

class AccuracySerializer(serializers.Serializer):
   """Your data serializer, define your fields here."""
   accuracy = serializers.IntegerField()