from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock


class PendulumBob(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PendulumCart(Widget):
    velocity = NumericProperty(0)

    def move(self):
        self.pos = Vector(velocity, 0) + self.pos


class SwirlWorld(Widget):
    cart = ObjectProperty(None)
    bob = ObjectProperty(None)
    
    def init(self):
        # self.cart.center = self.center
        # self.bob.center = self.center + (100, 100)
        pass

    def update(self, dt):
        # self.rk4()
        pass


class SwirlApp(App):
    def build(self):
        world = SwirlWorld()
        world.init()
        Clock.schedule_interval(world.update, 1.0 / 60.0)
        return world


if __name__ == '__main__':
    SwirlApp().run()
