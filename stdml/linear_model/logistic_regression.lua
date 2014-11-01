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

local log_reg,log_reg_methods = class("logistic_regression")

log_reg._NAME    = "stdml.linear_model.logistic_regression"
log_reg._VERSION = "0.1"

function log_reg.constructor(self,params)
  local params = get_table_fields({ l1 = { type_match="number" },
                                    l2 = { type_match="number" },
                                    fit_intercept = { type_match="boolean", default=true },
                                    -- class_weight = { },
                                    shuffle = { isa_match=random, default=random() },
                                    tol = { type_match="number", default=0.0001 },
                                    max_epochs = { type_match="number", default=1000 },
                                    min_epochs = { type_match="number", default=1 },
                                    verbose = { type_match="boolean" },
                                  }, params)
  self.params = params
end

local function train_ds(self,tr_data)
  local trainer = self.trainer
  local tr_loss = trainer:train_dataset(tr_data)
  return trainer,tr_loss
end

local function forward_matrix(self,x)
  local order = x:get_major_order()
  if order == "row_major" then x=x:clone("col_major") end
  local out = self.trainer:calculate(x)
  if order == "row_major" then return out:clone("row_major") end
  return out
end

function log_reg_methods:fit(x,y)
  assert(not class.is_a(x,matrix.sparse), "NOT IMPLEMENTED FOR SPARSE MATRICES")
  assert(class.is_a(x,matrix) or class.is_a(x,matrix.sparse),
         "Needs a matrix or matrix.sparse as 1st argument")
  assert(class.is_a(y,matrix) or class.is_a(y,matrixInt32) or type(y)=="table",
         "Needs a matrix, matrixInt32 or table as 2nd argument")
  if type(y)=="table" then y = matrix(y)
  elseif class.is_a(y,matrixInt32) then y = y:to_float()
  end
  local x_dim,y_dim = x:dim(),y:dim()
  assert(#x_dim == 2, "Needs a 2D matrix as 1st argument")
  assert(#y_dim == 1 or (#y_dim==2 and y_dim[2]==1),
         "Needs a vector as 2nd argument")
  local num_samples,num_features = x:dim(1),x:dim(2)
  local class_dict,num_classes = {},0
  local y = y:map(function(x)
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
  self.num_classes = num_classes
  -- MODEL
  local num_outputs = (num_classes==2) and 1 or num_classes
  local model = ann.components.stack()
  model:push( ann.components.dot_product{ input=num_features, output=num_outputs,
                                          weights="coef_" } )
  if self.params.fit_intercept then
    model:push( ann.components.bias{ size=num_outputs, weights="intercept_" } )
  end
  model:push( ann.components.actf.log_logistic() )
  --
  local loss = (num_classes==2) and ann.loss.cross_entropy() or ann.loss.multi_class_cross_entropy()
  local trainer = trainable.supervised_trainer(model, loss, num_samples,
                                               ann.optimizer.cg())
  trainer:build()
  local coef_,intercept_ = trainer:weights("coef_"),trainer:weights("intercept_")
  self.coef_,self.intercept_= coef_,intercept_
  self.trainer = trainer
  coef_:zeros()
  intercept_:zeros()
  --
  local l1,l2 = self.params.l1,self.params.l2
  if l1 then trainer:set_layerwise_option("coef_", l1) end
  if l2 then trainer:set_layerwise_option("coef_", l2) end
  --
  local pocket = trainable.train_wo_validation{
    percentage_stopping_criterion = self.params.tol,
    min_epochs = self.params.min_epochs,
    max_epochs = self.params.max_epochs,
  }
  if x:get_major_order() == "col_major" then x=x:clone("row_major") end
  if y:get_major_order() == "col_major" then y=y:clone("row_major") end
  local in_ds   = dataset.matrix(x)
  local out_ds  = dataset.matrix(y)
  if num_classes > 2 then
    out_ds = dataset.indexed(out_ds, { dataset.identity(num_classes) })
  end
  local tr_data = {
    input_dataset  = in_ds,
    output_dataset = out_ds,
    shuffle        = self.params.shuffle,
    replacement    = num_samples,
  }
  local verbose = self.params.verbose
  while pocket:execute(train_ds,self,tr_data) do
    if verbose then print(pocket:get_state_string()) end
  end
  return self
end

function log_reg_methods:predict(x)
  local out = forward_matrix(self,x)
  if self.num_classes > 2 then
    out:max(2)
  else
    out = out:exp():gt(0.5):to_float()
  end
  return out
end

function log_reg_methods:predict_log_proba(x)
  return forward_matrix(self,x)
end

function log_reg_methods:predict_proba(x)
  return forward_matrix(self,x):exp()
end

function log_reg_methods:transform(x,threshold)
  assert(self.coef_, "Needs a fitted model")
  local threshold = threshold or 0.0
  local gt_zero = matrix.op.abs(self.coef_):gt(threshold)
  local sel_cols = matrixBool(gt_zero:dim(2)):zeros()
  for i=1,gt_zero:dim(1) do
    sel_cols[{}] = sel_cols + gt_zero:select(1,i)
  end
  return x:index(2,sel_cols)
end

return log_reg
