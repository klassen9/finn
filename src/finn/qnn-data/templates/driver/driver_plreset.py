from pynq import MMIO

# TODO currently only for ZynqUS
reset_register_dict = {
    "ADDR_BASE": 0xFF0A_0000,
    "ADDR_RANGE": 0x1000,
    "RESET_BIT": 0x8000_0000, 
    "MASK_DATA_LSW" : 0x028,
    "MASK_DATA_MSW" : 0x02C,
    "DATA" : 0x054,
    "DATA_RO" : 0x074,
    "DIRM" : 0x344,
    "OEN" : 0x348,
    "INT_MASK" : 0x34C,
    "INT_EN" : 0x350,
    "INT_DIS" : 0x354,
}

class PLreset:
    def __init__(self, gpio_register):
        self._registers = gpio_register
        self._gpio = MMIO(self._registers["ADDR_BASE"], self._registers["ADDR_RANGE"])
        self._gpio.write(self._registers["INT_DIS"], self._registers["RESET_BIT"]) # Disable Interrupts
        self._gpio.write(self._registers["DIRM"], self._registers["RESET_BIT"]) # Enable Output
        
    def set_reset(self, reset=False):
        # We consider only the upper 16 bit
        if reset == False:
            data_lsw = 0
            data_msw = 0
        else:
            data_lsw = self._registers["RESET_BIT"] & 0x0000_FFFF
            data_msw = self._registers["RESET_BIT"] >> 16
        
        mask_lsw = (~self._registers["RESET_BIT"] & 0xFFFF) << 16
        mask_msw = ~self._registers["RESET_BIT"] & 0xFFFF_0000
        
        self._gpio.write(self._registers["MASK_DATA_LSW"], data_lsw | mask_lsw)
        self._gpio.write(self._registers["MASK_DATA_MSW"], data_msw | mask_msw)

    def triggerPLreset(self):
        if self.reset is None: 
            Exception("No reset defined") 
        self.reset.set_reset(False)
        self.reset.set_reset(True)