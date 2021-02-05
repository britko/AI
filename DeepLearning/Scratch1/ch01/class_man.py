class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbey(self):
        print("Good-bey " + self.name + "!")

m = Man("David")
m.hello()
m.goodbey()