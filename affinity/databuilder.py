import random

class Profile:
    def __init__(self, percentInterestInCommonlyViewedTag, percentInterestUnCommonlyViewedTag,  percent_interests, percent_not_interests, percent_interest_age_better_avg, percent_interest_age_worse_avg, percent_interest_on_active_days, percent_interest_on_non_active_days):
        self.percent_interests = percent_interests
        self.percentInterestInCommonlyViewedTag = percentInterestInCommonlyViewedTag
        self.percentInterestUnCommonlyViewedTag = percentInterestUnCommonlyViewedTag
        self.percent_not_interests = percent_not_interests
        self.percent_interest_age_better_avg = percent_interest_age_better_avg
        self.percent_interest_age_worse_avg = percent_interest_age_worse_avg
        self.percent_interest_on_active_days = percent_interest_on_active_days
        self.percent_interest_on_non_active_days = percent_interest_on_non_active_days

class ContentCtx:
    def __init__(self, isTagInterest, isInterestTopic, createDateNearAvg, isActiveDayOfWeek):
        self.isTagInterest = isTagInterest
        self.isInterestTopic = isInterestTopic
        self.createDateNearAvg = createDateNearAvg
        self.isActiveDayOfWeek = isActiveDayOfWeek

class Person:
    def __init__(self, profile):
        self.profile = profile

    def response(self, content):
        enjoy = 0
        hate = 0

        facts = []

        if(content.isTagInterest): #tag_ep_count percent enjoyment of content with this tag. like read or both. of total items seen with this tag
            result = self.feeling(self.profile.percentInterestInCommonlyViewedTag)
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.percentInterestUnCommonlyViewedTag)
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(0)

        if(content.isInterestTopic): #percent_interests %of time user reads likes things in interest
            result = self.feeling(self.profile.percent_interests)
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.percent_not_interests) #percent_not_interests %of time user reads likes things not in interest
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(0)

        if(content.createDateNearAvg):
            result = self.feeling(self.profile.percent_interest_age_better_avg) #percent of time a user reads/likes an article around his average
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.percent_interest_age_worse_avg) #percent of time a user reads/likes an article not near his average
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(0)

        if(content.isActiveDayOfWeek):
            result = self.feeling(self.profile.percent_interest_on_active_days) #percent of time a user reads/likes an article on active days
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(1)
        else:
            result = self.feeling(self.profile.percent_interest_on_non_active_days) #percent of time a user reads/likes an article on non active days
            enjoy = enjoy + result[0]
            hate = hate + result[1]
            facts.append(0)
        
        rates = [self.profile.percentInterestInCommonlyViewedTag, self.profile.percentInterestUnCommonlyViewedTag, self.profile.percent_interests, self.profile.percent_not_interests, self.profile.percent_interest_age_better_avg, self.profile.percent_interest_age_worse_avg, self.profile.percent_interest_on_active_days, self.profile.percent_interest_on_non_active_days]
        row = facts + rates
        if(enjoy > hate):
            row.append(1)
            return row
        elif(enjoy == hate):
            if(random.randint(0, 100) <= 60):
                row.append(1)
                return row
            else:
                row.append(0)
                return row
        else:
            row.append(0)
            return row

    def feeling(self, interestPercent):
        importance = abs(50 - interestPercent) 
        if(interestPercent > random.randint(0, 100)):
            return (importance, 0)
        else:
            return (0, importance)

people = []
tens = [10, 20, 30,40,50,60,70,80,90,100]
for tagIdx in tens:
    for interestIdx in tens:
        for avgAgeIdx in tens:
            for activeDayIdx in tens:
                tagSplit = (tagIdx, 100 - tagIdx)
                pI = (interestIdx, 100 - interestIdx)
                avgAge = (avgAgeIdx, 100 - avgAgeIdx)
                actDay = (activeDayIdx, 100 - activeDayIdx)
                people.append(Person(profile = Profile(percentInterestInCommonlyViewedTag = tagSplit[0], percentInterestUnCommonlyViewedTag=tagSplit[1], percent_interests=pI[0], percent_not_interests=pI[1], percent_interest_age_better_avg=avgAge[0], percent_interest_age_worse_avg=avgAge[1], percent_interest_on_active_days=actDay[0], percent_interest_on_non_active_days=actDay[1])))

print(len(people))
contentItems = []

for tag in range(2):
    for topic in range(2):
        for nearness in range(2):
            for activeDay in range(2):
                contentItems.append(ContentCtx(tag, topic, nearness, activeDay))


import csv

with open('contentAffinity.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['isTagInInterest', 'isTopicInInterest', 'isNearAvgCreateDate', 'isActiveDay', 'pctInterestCommonlyViewedTag', 'pctInterestUnCommonlyViewedTag', 'pctInterstInSelectedTopic', 'pctInterstInNonselectedTopic', 'pctInterestAvgCreateDateTime', 'pctInterestInStoriesCreatedNotNearAverage', 'pctInterestOnActiveDays', 'pctInterestOnNonActiveDays', 'enjoyed'])
    for lp in range(50):
        for content in contentItems:
            for person in people:
                filewriter.writerow(person.response(content))