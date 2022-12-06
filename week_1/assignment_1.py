from dataclasses import dataclass

@dataclass
class Mountain:
    name : str
    elevation : int
    def to_str(self) :       #it will first convert the parameters to into string and return string as well 
        #self is used to access the fields of class
        print("Mountain Name is ",self.name, "and elevation is ", str(self.elevation), " feet")
    
mountain_1 = Mountain("Mount Everest", 29029) # creating an instance of Mountain class
mountain_1.to_str() # accessing to_str method of Mountain class 

