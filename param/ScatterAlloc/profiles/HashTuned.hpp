/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Carlchristian Eckert, Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.

/** @file
 *
 * A retuned scatter hashing profile (different primes than HashDefault),
 * a dependent template on the heap config that ignores it.
 */

#pragma once

namespace picongpu
{
    template<class /*T_HeapConfig*/>
    struct HashTuned
    {
        static constexpr auto hashingK = 6559797u;
        static constexpr auto hashingDistMP = 22259u;
        static constexpr auto hashingDistWP = 1u;
        static constexpr auto hashingDistWPRel = 1u;
    };

} // namespace picongpu
