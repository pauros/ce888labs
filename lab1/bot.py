from sopel import module
from emo.wdemotions import EmotionDetector
a = 0.01
a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0


emo = EmotionDetector()
@module.rule('')

#,a2,a3,a4,a5,a6,i

def hi(bot, trigger):
    
    global ave,a1,a2,a3,a4,a5,a6
    print(trigger, trigger.nick)
    #bot.say('Hello guys, ' + trigger.nick)
    emotion=emo.detect_emotion_in_raw_np(trigger)
    print emotion
    
    a1= a1+a*(emotion[0]-a1)
    a2= a2+a*(emotion[1]-a1)
    a3= a3+a*(emotion[2]-a1)
    a4= a4+a*(emotion[3]-a1)
    a5= a5+a*(emotion[4]-a1)
    a6= a6+a*(emotion[5]-a1)
    print a1,a2,a3,a4,a5,a6
