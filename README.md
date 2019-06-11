# Machine Learning for Flash

## Models
* **Notifications** notifications.tflite
* **Affinity** affinity.tflite

### Notifications

The notifications model predicts whether or not a user will open a notification based on observed behavior.

#### Inputs 
The inputs are mix of User observed behaviors, explicitly set preferences and characteristics of the incoming notification.

**isActiveDayOfWeek** = 0,1  *true (1) if the today is typically an active day for the user*
**isInInterest** = 0,1 *true if the notification*
**isAM** = 0,1 *notification time am or pm*
**isGeneric** = 0,1 *the notification has the generic message or a headline*

These are open rate percentages (OPR) each characteristic of a notification.
**amOPR** 
**pmOPR** 
**activeDayOPR** 
**nonActiveDayOPR** 
**genericDescriptionOPR** 
**uniqueDescriptionOPR** 
**interestOPR** 
**otherTopicOPR** 

#### Output
**opened** = 0,1

I trained Tensorflow's Deep Neural Network Classifier algorithm to create the model.  The script data builder creates simulated notification responses that includes 6561 people, 16 unique Notifications each sent to the user 10 times with a final output of 516982 opens & 532778 ignored or dismissed notifications for a total of 1,049,760 rows of response.

For users that are opt-in to notifications we could use this to model to filter notifications on the client, sending them only when they are most relevant.
