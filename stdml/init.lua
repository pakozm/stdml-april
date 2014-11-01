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

local stdml = {
  _NAME = "stdml",
  _VERSION = "0.1",
}

stdml.linear_model = require "stdml.linear_model"

return stdml
