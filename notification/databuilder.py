import random 

class Profile:
    def __init__(self, am, pm, descriptionGeneric, descriptionSpecific, percentActiveDays, percentNonActiveDays, interestOpenRate, nonInterestOpenRate):
        self.am = am
        self.pm = pm
        self.descriptionGeneric = descriptionGeneric
        self.descriptionSpecific = descriptionSpecific
        self.percentActiveDays = percentActiveDays
        self.percentNonActiveDays = percentNonActiveDays
        self.nonInterestOpenRate = nonInterestOpenRate
        self.interestOpenRate = interestOpenRate
        
class Notification:
    def __init__(self, isAM, isActiveDayOfWeek, isGeneric, isTopicInInterests):
        self.isAM = isAM
        self.isActiveDayOfWeek = isActiveDayOfWeek
        self.isGeneric = isGeneric
        self.isTopicInInterests = isTopicInInterests
        
class Person:
    def __init__(self, profile):
        self.profile = profile
        
    def response(self, notification):
        positive = 0
        negative = 0
        
        facts = []
        
        if(notification.isAM):
            result = self.feeling(self.profile.am)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.pm)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(0)
        
        if(notification.isActiveDayOfWeek):
            result = self.feeling(self.profile.percentActiveDays)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.percentNonActiveDays)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(0)
            
        if(notification.isGeneric):
            result = self.feeling(self.profile.descriptionGeneric)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.descriptionSpecific)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(0)
            
        if(notification.isTopicInInterests):
            result = self.feeling(self.profile.interestOpenRate)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.nonInterestOpenRate)
            positive = positive + result[0]
            negative = negative + result[1]
            facts.append(0)
            
        rates = [self.profile.am, self.profile.pm, self.profile.percentActiveDays, self.profile.percentNonActiveDays, self.profile.descriptionGeneric, self.profile.descriptionSpecific, self.profile.interestOpenRate, self.profile.nonInterestOpenRate]
        
        row = facts + rates
        
        if(positive > negative):
            row.append(NotificationResponse.OPEN)
            return row
        elif(positive == negative):
            if(random.randint(0,100) > 50):
                row.append(NotificationResponse.OPEN)
                return row 
            else:
                row.append(NotificationResponse.CLOSE)
                return row
        else:
            row.append(NotificationResponse.CLOSE)
            return row
        
    def feeling(self, interestPercent):
        importance = abs(50 - interestPercent) 
            
        if(interestPercent > random.randint(0, 100)):
            return (importance, 0)
        else:
            return (0, importance)

class NotificationResponse:
    OPEN = 1
    CLOSE = 0

people = []
buckets = [25, 50, 75]

for interest in buckets:
    for nonInterest in buckets:
        for am in buckets:
            for pm in buckets:
                for generic in buckets:
                    for specific in buckets:
                        for activeDays in buckets:
                            for nonActiveDays in buckets:
                                people.append(Person(profile = Profile(am=am,pm=pm, descriptionGeneric=generic, descriptionSpecific=specific, percentActiveDays = activeDays, percentNonActiveDays = nonActiveDays, interestOpenRate = interest, nonInterestOpenRate=nonInterest)))

print("people")
print(len(people))

notifications = []

for isAM in range(2):
    for isActiveDay in range(2):
        for isGeneric in range(2):
            for isTopicInInterests in range(2):
                notifications.append(Notification(isAM == 1, isActiveDay == 1, isGeneric == 1, isTopicInInterests == 1))

print("notifications")
print(len(notifications))
import csv

NUM_NOTIFICATIONS_TO_SEND = 10

ignores = 0
opens = 0

with open('raw/notifications.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['isAm', 'isActiveDayOfWeek', 'isGeneric', 'isInInterest','amOPR', 'pmOPR', 'activeDayOPR', 'nonActiveDayOPR', 'genericDescriptionOPR', 'uniqueDescriptionOPR', 'interestOPR', 'otherTopicOPR', 'opened'])
    for lp in range(NUM_NOTIFICATIONS_TO_SEND):
        if lp % 5 == 0 and lp > 0:
            print(lp)
        for notification in notifications:
            for person in people:
                response = person.response(notification)
                if(response[12] == NotificationResponse.CLOSE):
                    ignores+=1
                else: 
                    opens+=1
                filewriter.writerow(response)

print("opens")
print(opens)
print("dismss/ignores")
print(ignores)

