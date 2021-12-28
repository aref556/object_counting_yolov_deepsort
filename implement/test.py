class Test:
    def __init__(self):
        self.x = 0
        f = {}
        f['{}'.format(1)] = 5
        f['{}'.format(2)] = 6
        self.xx = 5
        for key in f:
            setattr(self, key, f[key])
        
t1 = Test()

print(t1)

t = {}
print(t.get(3))
t[1] = 3
print(t)