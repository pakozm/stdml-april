--[[
  Copyright (c) 2014 Francisco Zamora-Martinez (pakozm@gmail.com)

  The STDML is an extension for APRIL-ANN toolkit, and both are free software;
  you can redistribute it and/or modify it under the terms of the GNU General
  Public License version 3 as published by the Free Software Foundation
  
  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with
  this library; if not, write to the Free Software Foundation, Inc., 59 Temple
  Place, Suite 330, Boston, MA 02111-1307 USA
]]


local class_util = {}

function class_util.check_target_classes(y)
  local num_classes,class_dict = 0,{}
  y:map(function(x)
      assert(math.floor(x)==x, "Need integer numbers as class identifiers")
      if not class_dict[x] then num_classes=num_classes+1 class_dict[x]=true end
  end)
  if num_classes == 2 then
    assert(y:min() == 0 and y:max() == 1,
           "For 2-class problems use 0/1 targets")
  else
    assert(y:min() == 1 and y:max() == num_classes,
           "For multi-class problems use 1..N targets")
  end
  return num_classes,class_dict
end

return class_util
